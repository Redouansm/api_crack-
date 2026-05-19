import os
import torch
import torchvision.models as tv_models
import torch.nn as nn

# =====================================
# SAFE OPENVINO IMPORT
# =====================================

OPENVINO_AVAILABLE = False

try:

    from openvino.runtime import Core

    OPENVINO_AVAILABLE = True

except Exception:

    OPENVINO_AVAILABLE = False

# =====================================
# MODEL CONFIG
# =====================================

MODEL_NAME = "mobilenetv2_best"

# =====================================
# CACHE
# =====================================

_MODELS = {}

# =====================================
# DEVICE DETECTION
# =====================================

def get_best_device():

    # NVIDIA GPU
    if torch.cuda.is_available():

        print(
            "Using NVIDIA GPU (CUDA)"
        )

        return "cuda"

    # OPENVINO
    if OPENVINO_AVAILABLE:

        try:

            core = Core()

            devices = (
                core.available_devices
            )

            print(
                "OpenVINO devices:",
                devices
            )

            if "GPU" in devices:

                print(
                    "Using Intel iGPU"
                )

                return "openvino_gpu"

            return "openvino_cpu"

        except Exception as e:

            print(
                "OpenVINO error:",
                e
            )

    print(
        "Using Torch CPU"
    )

    return "cpu"

# =====================================
# LOAD TORCH MODEL
# =====================================

def load_torch_model(
    weights_path,
    device
):

    criticality_model = tv_models.mobilenet_v2(
        weights=None
    )

    criticality_model.classifier[1] = nn.Linear(
        criticality_model.classifier[1].in_features,
        3
    )

    checkpoint = torch.load(

        weights_path,

        map_location=device,

        weights_only=False
    )

    # =====================================
    # EXTRACT STATE DICT
    # =====================================

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            checkpoint = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            checkpoint = checkpoint[
                "model_state_dict"
            ]

    # =====================================
    # REMOVE module.
    # =====================================

    cleaned = {}

    for k, v in checkpoint.items():

        if k.startswith(
            "module."
        ):

            k = k.replace(
                "module.",
                ""
            )

        cleaned[k] = v

    criticality_model.load_state_dict(
        cleaned
    )

    criticality_model.eval()

    if device == "cuda":

        criticality_model.half()

        criticality_model.to("cuda")

    return criticality_model

# =====================================
# LOAD MODEL
# =====================================

def load_model(model_name):

    global _MODELS

    if model_name in _MODELS:

        return _MODELS[model_name]

    # =====================================
    # PATHS
    # =====================================

    current_dir = os.path.dirname(
        __file__
    )

    app_dir = os.path.dirname(
        current_dir
    )

    model_dir = os.path.join(

        app_dir,

        "models",

        "crack"
    )

    weights_path = os.path.join(

        model_dir,

        f"{MODEL_NAME}.pth"
    )

    xml_path = os.path.join(

        model_dir,

        f"{MODEL_NAME}.xml"
    )

    bin_path = os.path.join(

        model_dir,

        f"{MODEL_NAME}.bin"
    )

    device_type = get_best_device()

    # =====================================
    # CUDA
    # =====================================

    if device_type == "cuda":

        model = load_torch_model(

            weights_path,

            "cuda"
        )

        result = {

            "backend": "cuda",

            "model": model
        }

    # =====================================
    # OPENVINO
    # =====================================

    elif device_type in [

        "openvino_gpu",

        "openvino_cpu"
    ]:

        try:

            core = Core()

            ov_model = core.read_model(

                model=xml_path,

                weights=bin_path
            )

            device_name = (

                "GPU"

                if device_type == "openvino_gpu"

                else "CPU"
            )

            compiled_model = core.compile_model(

                model=ov_model,

                device_name=device_name,

                config={

                    "PERFORMANCE_HINT": "THROUGHPUT"
                }
            )

            result = {

                "backend": "openvino",

                "model": compiled_model
            }

        except Exception as e:

            print(
                "OpenVINO failed:",
                e
            )

            model = load_torch_model(

                weights_path,

                "cpu"
            )

            result = {

                "backend": "cpu",

                "model": model
            }

    # =====================================
    # CPU
    # =====================================

    else:

        model = load_torch_model(

            weights_path,

            "cpu"
        )

        result = {

            "backend": "cpu",

            "model": model
        }

    _MODELS[model_name] = result

    print(
        "Backend loaded:",
        result["backend"]
    )

    return result