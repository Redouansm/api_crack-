import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ai.device import LazyModel, load_torch_or_openvino, model_path

MODEL_NAME = os.getenv("CRACK_MODEL_NAME", "best_model_b3")

INPUT_SIZE = 448
THRESHOLD  = float(os.getenv("CRACK_THRESHOLD", "0.95"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GPU_PATCH_SIZE     = int(os.getenv("CRACK_GPU_PATCH_SIZE", "448"))
GPU_OVERLAP        = float(os.getenv("CRACK_GPU_OVERLAP", "0.3"))
GPU_USE_TTA        = os.getenv("CRACK_GPU_TTA", "1") == "1"
GPU_BATCH_SIZE     = int(os.getenv("CRACK_GPU_BATCH", "8"))
GPU_THRESHOLD      = float(os.getenv("CRACK_GPU_THRESHOLD", str(THRESHOLD)))
GPU_MORPHO_CLOSE   = int(os.getenv("CRACK_GPU_MORPHO_CLOSE", "3"))
GPU_MIN_CRACK_AREA = int(os.getenv("CRACK_GPU_MIN_AREA", "50"))


def build_architecture():
    import segmentation_models_pytorch as smp
    return smp.Segformer(encoder_name="mit_b3", encoder_weights=None, in_channels=3, classes=1)


def _load():
    data = load_torch_or_openvino(
        build_architecture,
        weights_path=model_path("crack", f"{MODEL_NAME}.pth"),
        xml_path=model_path("crack", f"{MODEL_NAME}.xml"),
    )
    print(f"[crack] backend = {data['backend']}")
    if data["backend"] == "cuda":
        print(
            f"[crack] mode GPU HD (patch={GPU_PATCH_SIZE}, overlap={GPU_OVERLAP}, "
            f"TTA={GPU_USE_TTA}, batch={GPU_BATCH_SIZE}, threshold={GPU_THRESHOLD})"
        )
    return data


CRACK_MODEL = LazyModel("crack", _load)


# ─────────────────────────────────────────────────────────────────────────────
# MODE RAPIDE (CPU / OpenVINO) : 448×448 avec padding
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(image_rgb):
    h, w   = image_rgb.shape[:2]
    scale  = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded  = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    x = (padded.astype(np.float32) / 255.0 - MEAN) / STD
    x = np.expand_dims(np.transpose(x, (2, 0, 1)), axis=0)
    return x, h, w, new_h, new_w


def predict_fast(image_rgb, data):
    backend, model = data["backend"], data["model"]
    x, h, w, new_h, new_w = _preprocess(image_rgb)

    if backend == "openvino":
        pred = list(model([x]).values())[0].squeeze()
        pred = 1 / (1 + np.exp(-pred))
    elif backend == "cuda":
        with torch.inference_mode():
            pred = torch.sigmoid(model(torch.from_numpy(x).half().to("cuda"))).squeeze().float().cpu().numpy()
    else:
        with torch.inference_mode():
            pred = torch.sigmoid(model(torch.from_numpy(x).float())).squeeze().cpu().numpy()

    pred_bin = (pred[:new_h, :new_w] > THRESHOLD).astype(np.uint8)
    return cv2.resize(pred_bin, (w, h), interpolation=cv2.INTER_NEAREST)


_TTA_OPS = [
    (lambda t: t,                              lambda t: t),
    (lambda t: torch.flip(t, dims=[3]),        lambda t: torch.flip(t, dims=[3])),
    (lambda t: torch.flip(t, dims=[2]),        lambda t: torch.flip(t, dims=[2])),
    (lambda t: torch.rot90(t, 1, dims=[2, 3]), lambda t: torch.rot90(t, -1, dims=[2, 3])),
    (lambda t: torch.rot90(t, 2, dims=[2, 3]), lambda t: torch.rot90(t, -2, dims=[2, 3])),
    (lambda t: torch.rot90(t, 3, dims=[2, 3]), lambda t: torch.rot90(t, -3, dims=[2, 3])),
]


def _model_probs(batch, model, out_size):
    logits = model(batch)
    if logits.shape[-2:] != (out_size, out_size):
        logits = F.interpolate(logits.float(), size=(out_size, out_size), mode="bilinear", align_corners=False)
    return torch.sigmoid(logits.float())


def _patch_positions(length, patch, stride):
    pos = list(range(0, length - patch + 1, stride))
    if pos[-1] + patch < length:
        pos.append(length - patch)
    return pos


@torch.inference_mode()
def _predict_prob_hd_gpu(image_rgb, model, device="cuda", dtype=torch.float16):
    H, W = image_rgb.shape[:2]
    P    = GPU_PATCH_SIZE

    img = torch.from_numpy(np.ascontiguousarray(image_rgb)).to(device)
    img = img.permute(2, 0, 1).float().div_(255.0)

    pad_h, pad_w = max(0, P - H), max(0, P - W)
    if pad_h or pad_w:
        img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
    Hp, Wp = img.shape[1:]

    mean = torch.tensor(MEAN, device=device).view(3, 1, 1)
    std  = torch.tensor(STD,  device=device).view(3, 1, 1)
    img  = ((img - mean) / std).to(dtype)

    stride = max(1, int(P * (1 - GPU_OVERLAP)))
    coords = [(y, x) for y in _patch_positions(Hp, P, stride) for x in _patch_positions(Wp, P, stride)]

    prob_sum = torch.zeros((Hp, Wp), device=device, dtype=torch.float32)
    count    = torch.zeros((Hp, Wp), device=device, dtype=torch.float32)
    ops      = _TTA_OPS if GPU_USE_TTA else _TTA_OPS[:1]

    for i in range(0, len(coords), GPU_BATCH_SIZE):
        chunk = coords[i:i + GPU_BATCH_SIZE]
        batch = torch.stack([img[:, y:y + P, x:x + P] for (y, x) in chunk])

        probs = torch.zeros((len(chunk), 1, P, P), device=device, dtype=torch.float32)
        for fwd, inv in ops:
            probs += inv(_model_probs(fwd(batch), model, P))
        probs /= len(ops)

        for k, (y, x) in enumerate(chunk):
            prob_sum[y:y + P, x:x + P] += probs[k, 0]
            count[y:y + P, x:x + P]    += 1

    return (prob_sum / count.clamp_min(1))[:H, :W].cpu().numpy()


def _postprocess(prob_map):
    pred = (prob_map > GPU_THRESHOLD).astype(np.uint8)

    if GPU_MORPHO_CLOSE > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GPU_MORPHO_CLOSE, GPU_MORPHO_CLOSE))
        pred   = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

    if GPU_MIN_CRACK_AREA > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        keep = np.zeros(n, dtype=np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= GPU_MIN_CRACK_AREA
        pred = keep[labels]

    return pred


def predict_crack_mask(image_rgb: np.ndarray) -> np.ndarray:

    data = CRACK_MODEL.get()

    with CRACK_MODEL.lock:
        if data["backend"] != "cuda":
            return predict_fast(image_rgb, data)
        try:
            return _postprocess(_predict_prob_hd_gpu(image_rgb, data["model"]))
        except torch.cuda.OutOfMemoryError:
            print("[crack] GPU OOM en mode HD -> repli sur le mode rapide (448)")
            torch.cuda.empty_cache()
            return predict_fast(image_rgb, data)
