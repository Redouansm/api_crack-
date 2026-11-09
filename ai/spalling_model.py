import os
import re
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ai.device import LazyModel, clean_state_dict, model_path, torch_device

WEIGHTS   = os.getenv("SPALLING_WEIGHTS", model_path("spalling", "best_segformer_b2_v2.pth"))
IMG_SIZE  = int(os.getenv("SPALLING_IMG_SIZE", "512"))   # taille d'entraînement
THRESHOLD = float(os.getenv("SPALLING_THRESHOLD", "0.5"))  # 0.5 = argmax du notebook
USE_TTA   = os.getenv("SPALLING_TTA", "1") == "1"         # flip horizontal (evaluate(tta=True))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ID2LABEL = {0: "background", 1: "spalling"}

_PATCH = [r"segformer\.encoder\.patch_embeddings\.(\d+)\.proj\.weight$",
          r"segformer\.stages\.(\d+)\.patch_embeddings\.proj\.weight$"]
_BLOCK = [r"segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)$",
          r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.(.+)$"]
_SR    = re.compile(r"attention\.(self\.sr|sequence_reduction\.sequence_reduction)\.weight$")
_FC1   = re.compile(r"mlp\.(dense1|fc1)\.weight$")


def _config_from_state_dict(sd: dict):
    """Reconstruit la SegformerConfig (mit-b2) à partir des formes des tenseurs."""
    from transformers import SegformerConfig

    patch = {}
    for k, v in sd.items():
        for pat in _PATCH:
            if m := re.match(pat, k):
                patch[int(m.group(1))] = v
    if not patch:
        raise ValueError("Fichier de poids non reconnu comme un SegFormer (transformers)")
    n_stages = len(patch)

    depths, sr_ratios, mlp_hidden = [0] * n_stages, [1] * n_stages, [None] * n_stages
    for k, v in sd.items():
        for pat in _BLOCK:
            if m := re.match(pat, k):
                i, j, rest = int(m.group(1)), int(m.group(2)), m.group(3)
                depths[i] = max(depths[i], j + 1)
                if j == 0 and _SR.search(rest):
                    sr_ratios[i] = int(v.shape[-1])
                if j == 0 and _FC1.search(rest):
                    mlp_hidden[i] = int(v.shape[0])

    hidden_sizes = [int(patch[i].shape[0]) for i in range(n_stages)]
    classifier   = sd["decode_head.classifier.weight"]
    num_labels   = int(classifier.shape[0])

    return SegformerConfig(
        num_channels=3,
        num_encoder_blocks=n_stages,
        depths=depths,
        sr_ratios=sr_ratios,
        hidden_sizes=hidden_sizes,
        patch_sizes=[int(patch[i].shape[-1]) for i in range(n_stages)],
        strides=[4] + [2] * (n_stages - 1),
        num_attention_heads=[1, 2, 5, 8][:n_stages],   # identique pour toute la famille MiT
        mlp_ratios=[(mlp_hidden[i] or 4 * hidden_sizes[i]) // hidden_sizes[i] for i in range(n_stages)],
        decoder_hidden_size=int(classifier.shape[1]),
        num_labels=num_labels,
        id2label=ID2LABEL if num_labels == 2 else None,
        label2id={v: k for k, v in ID2LABEL.items()} if num_labels == 2 else None,
    )


def _load():
    from safetensors.torch import save_file
    from transformers import SegformerForSemanticSegmentation

    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"Poids SegFormer introuvables : {WEIGHTS}")

    device = torch_device()
    sd     = clean_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False))
    config = _config_from_state_dict(sd)

    with tempfile.TemporaryDirectory() as tmp:
        config.save_pretrained(tmp)
        save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(tmp, "model.safetensors"))
        model, info = SegformerForSemanticSegmentation.from_pretrained(tmp, output_loading_info=True)

    problems = {k: list(v) for k, v in info.items() if k != "error_msgs" and v}
    if problems:
        raise RuntimeError(f"Poids SegFormer incompatibles : {problems}")

    model.eval().to(device)
    if device == "cuda":
        model.half()

    print(f"[spalling] SegFormer (hidden={config.hidden_sizes}, depths={config.depths}, "
          f"decoder={config.decoder_hidden_size}) sur {device.upper()}, TTA={USE_TTA}")
    return {"model": model, "device": device}


SPALLING_MODEL = LazyModel("spalling", _load)


@torch.inference_mode()
def predict_spalling_prob(image_rgb: np.ndarray) -> np.ndarray:
    """Carte de probabilité (H, W) float32 de la classe 'spalling', à la taille d'origine."""

    m      = SPALLING_MODEL.get()
    model  = m["model"]
    device = m["device"]
    dtype  = torch.float16 if device == "cuda" else torch.float32
    H, W   = image_rgb.shape[:2]

    # Même redimensionnement que A.Resize (cv2.INTER_LINEAR sur uint8)
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    x = ((x - MEAN) / STD).to(device=device, dtype=dtype)

    with SPALLING_MODEL.lock:
        logits = model(pixel_values=x).logits
        if USE_TTA:
            logits = (logits + model(pixel_values=x.flip(-1)).logits.flip(-1)) / 2

    logits = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False)
    return torch.softmax(logits, dim=1)[0, 1].cpu().numpy()


def predict_spalling_mask(image_rgb: np.ndarray) -> tuple:
    prob = predict_spalling_prob(image_rgb)
    return (prob > THRESHOLD).astype(np.uint8), prob
