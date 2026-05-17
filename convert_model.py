import torch
import openvino as ov
import segmentation_models_pytorch as smp

# =====================================
# MODEL NAME
# =====================================

MODEL_NAME = "best_model_b3"

# =====================================
# LOAD MODEL
# =====================================

model = smp.Segformer(

    encoder_name="mit_b3",

    encoder_weights=None,

    in_channels=3,

    classes=1
)

# =====================================
# LOAD CHECKPOINT
# =====================================

checkpoint = torch.load(

    f"models/crack/{MODEL_NAME}.pth",

    map_location="cpu",

    weights_only=False
)

# =====================================
# EXTRACT STATE DICT
# =====================================

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        checkpoint = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        checkpoint = checkpoint[
            "state_dict"
        ]

# =====================================
# REMOVE module.
# =====================================

cleaned_state_dict = {}

for key, value in checkpoint.items():

    if key.startswith(
        "module."
    ):

        key = key.replace(
            "module.",
            ""
        )

    cleaned_state_dict[
        key
    ] = value

# =====================================
# LOAD WEIGHTS
# =====================================

model.load_state_dict(
    cleaned_state_dict
)

model.eval()

# =====================================
# EXAMPLE INPUT
# =====================================

example_input = torch.randn(

    1,

    3,

    448,

    448
)

# =====================================
# CONVERT OPENVINO
# =====================================

ov_model = ov.convert_model(

    model,

    example_input=example_input
)

# =====================================
# SAVE MODEL
# =====================================

ov.save_model(

    ov_model,

    f"models/crack/{MODEL_NAME}.xml"
)

print(
    "OpenVINO export DONE"
)