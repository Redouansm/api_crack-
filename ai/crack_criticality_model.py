import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from ai.device import LazyModel, load_torch_or_openvino, model_path

MODEL_NAME  = os.getenv("CRACK_CRITICALITY_MODEL_NAME", "mobilenetv2_best")
N_AUG       = int(os.getenv("CRACK_CRITICALITY_TTA", "10"))
CLASS_NAMES = {0: "FAIBLE", 1: "MOYEN", 2: "CRITIQUE"}

_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TTA_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def build_architecture():
    import torchvision.models as tv_models
    model = tv_models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    return model


def _load():
    data = load_torch_or_openvino(
        build_architecture,
        weights_path=model_path("crack", f"{MODEL_NAME}.pth"),
        xml_path=model_path("crack", f"{MODEL_NAME}.xml"),
    )
    print(f"[crack_criticality] backend = {data['backend']}")
    return data


CRITICALITY_MODEL = LazyModel("crack_criticality", _load)


def _softmax_np(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _probs(batch: torch.Tensor, data) -> np.ndarray:
    backend, model = data["backend"], data["model"]

    if backend == "openvino":
        # Le modèle OpenVINO a été exporté avec un batch de 1
        out = [list(model([b.unsqueeze(0).numpy()]).values())[0].squeeze() for b in batch]
        return np.stack([_softmax_np(o) for o in out])

    with torch.inference_mode():
        if backend == "cuda":
            batch = batch.half().to("cuda")
        return torch.softmax(model(batch), dim=1).float().cpu().numpy()


def predict_criticite(image, n_aug: int = N_AUG) -> dict:

    data  = CRITICALITY_MODEL.get()
    batch = torch.stack([TTA_TRANSFORM(image) for _ in range(n_aug)] + [VAL_TRANSFORM(image)])

    with CRITICALITY_MODEL.lock:
        probs = _probs(batch, data)

    final = probs.mean(axis=0)
    idx   = int(final.argmax())

    return {
        "criticite":  CLASS_NAMES[idx],
        "confidence": round(float(final[idx]), 3),
    }
