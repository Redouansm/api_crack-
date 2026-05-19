import torch
import openvino as ov
import segmentation_models_pytorch as smp
import torchvision.models as tv_models
import torch.nn as nn

# =====================================
# MODEL NAMES
# =====================================

MODEL_NAME        = "best_model_b3"
CRITICALITY_MODEL = "mobilenetv2_best"

# =====================================
# LOAD SEGFORMER ARCHITECTURE
# =====================================

model = smp.Segformer(

    encoder_name="mit_b3",

    encoder_weights=None,

    in_channels=3,

    classes=1
)

# =====================================
# LOAD MOBILENETV2 ARCHITECTURE
# =====================================

criticality_model = tv_models.mobilenet_v2(weights=None)

criticality_model.classifier[1] = nn.Linear(
    criticality_model.classifier[1].in_features,
    3  # Haute / Modérée / Très Haute
)

# =====================================
# LOAD CHECKPOINTS
# =====================================

checkpoint = torch.load(

    f"models/crack/{MODEL_NAME}.pth",

    map_location="cpu",

    weights_only=False
)

checkpoint_crit = torch.load(

    f"models/crack/{CRITICALITY_MODEL}.pth",

    map_location="cpu",

    weights_only=False
)

# =====================================
# EXTRACT STATE DICT — SEGFORMER
# =====================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        checkpoint = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        checkpoint = checkpoint[
            "state_dict"
        ]

# =====================================
# EXTRACT STATE DICT — MOBILENETV2
# =====================================

if isinstance(checkpoint_crit, dict):

    if "model_state_dict" in checkpoint_crit:

        checkpoint_crit = checkpoint_crit[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint_crit:

        checkpoint_crit = checkpoint_crit[
            "state_dict"
        ]

# =====================================
# REMOVE module. — SEGFORMER
# =====================================

cleaned_state_dict = {}

for key, value in checkpoint.items():

    if key.startswith("module."):

        key = key.replace(
            "module.",
            ""
        )

    cleaned_state_dict[key] = value

# =====================================
# REMOVE module. — MOBILENETV2
# =====================================

cleaned_crit = {}

for key, value in checkpoint_crit.items():

    if key.startswith("module."):

        key = key.replace(
            "module.",
            ""
        )

    cleaned_crit[key] = value

# =====================================
# LOAD WEIGHTS — SEGFORMER
# =====================================

model.load_state_dict(cleaned_state_dict)

model.eval()

# =====================================
# LOAD WEIGHTS — MOBILENETV2
# =====================================

criticality_model.load_state_dict(cleaned_crit)

criticality_model.eval()

# =====================================
# EXAMPLE INPUTS
# =====================================

example_input = torch.randn(
    1,
    3,
    448,  # SegFormer input size
    448
)

example_input_crit = torch.randn(
    1,
    3,
    224,  # MobileNetV2 input size
    224
)

# =====================================
# CONVERT OPENVINO — SEGFORMER
# =====================================

print("Converting SegFormer to OpenVINO...")

ov_model = ov.convert_model(

    model,

    example_input=example_input
)

# =====================================
# SAVE — SEGFORMER
# =====================================

ov.save_model(

    ov_model,

    f"models/crack/{MODEL_NAME}.xml"
)

print(f"SegFormer OpenVINO export DONE → models/crack/{MODEL_NAME}.xml")

# =====================================
# CONVERT OPENVINO — MOBILENETV2
# =====================================

print("Converting MobileNetV2 to OpenVINO...")

ov_model_crit = ov.convert_model(

    criticality_model,

    example_input=example_input_crit
)

# =====================================
# SAVE — MOBILENETV2
# =====================================

ov.save_model(

    ov_model_crit,

    f"models/crack/{CRITICALITY_MODEL}.xml"
)

print(f"MobileNetV2 OpenVINO export DONE → models/crack/{CRITICALITY_MODEL}.xml")