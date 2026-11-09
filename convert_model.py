import openvino as ov
import torch

from ai import crack_criticality_model, crack_model
from ai.device import load_torch_weights, model_path

EXPORTS = [
    # (nom, constructeur d'architecture, nom des poids, taille d'entrée)
    ("SegFormer mit-b3", crack_model.build_architecture,             crack_model.MODEL_NAME,             448),
    ("MobileNetV2",      crack_criticality_model.build_architecture, crack_criticality_model.MODEL_NAME, 224),
]


for label, build_fn, name, size in EXPORTS:
    print(f"Conversion {label} -> OpenVINO ...")

    model    = load_torch_weights(build_fn(), model_path("crack", f"{name}.pth"), "cpu")
    ov_model = ov.convert_model(model, example_input=torch.randn(1, 3, size, size))
    out      = model_path("crack", f"{name}.xml")
    ov.save_model(ov_model, out)

    print(f"  OK -> {out}")
