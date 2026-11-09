from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image

import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

import openvino as ov

import numpy as np
import os

# =====================================
# FASTAPI
# =====================================

app = FastAPI()

# =====================================
# OUTPUT FOLDER
# =====================================

os.makedirs("outputs", exist_ok=True)

# Permet accès images depuis navigateur
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)

# =====================================
# OPENVINO DEVICE
# =====================================

core = ov.Core()

print("Available devices:")
print(core.available_devices)

device_name = "GPU" if "GPU" in core.available_devices else "CPU"

print("Using device:", device_name)

# =====================================
# LOAD MODEL
# =====================================

model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

checkpoint = torch.load(
    "model/best_model_resnet101.pth",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("✅ PyTorch model loaded")

# =====================================
# CONVERT TO OPENVINO
# =====================================

dummy_input = torch.randn(1, 3, 512, 512)

ov_model = ov.convert_model(
    model,
    example_input=dummy_input
)

# =====================================
# COMPILE MODEL FOR iGPU
# =====================================

compiled_model = core.compile_model(
    model=ov_model,
    device_name=device_name,
    config={
        "PERFORMANCE_HINT": "LATENCY"
    }
)

print("✅ OpenVINO model compiled")

# Output layer
output_layer = compiled_model.output(0)

# =====================================
# IMAGE TRANSFORM
# =====================================

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# =====================================
# ROOT ROUTE
# =====================================

@app.get("/")
def home():

    return {
        "message": f"FastAPI segmentation API is running on {device_name}"
    }

# =====================================
# PREDICT ROUTE
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # =====================================
    # IMAGE NAME
    # =====================================

    filename = file.filename

    base_name = os.path.splitext(filename)[0]

    extension = os.path.splitext(filename)[1]

    # =====================================
    # LOAD IMAGE
    # =====================================

    image = Image.open(file.file).convert("RGB")

    original_size = image.size

    # =====================================
    # PREPROCESSING
    # =====================================

    input_tensor = transform(image).unsqueeze(0)

    # Tensor -> numpy
    input_numpy = input_tensor.numpy()

    # =====================================
    # OPENVINO INFERENCE
    # =====================================

    result = compiled_model([input_numpy])

    output = result[output_layer]

    # =====================================
    # SIGMOID
    # =====================================

    output = 1 / (1 + np.exp(-output))

    # =====================================
    # CREATE MASK
    # =====================================

    mask = output.squeeze()

    mask = (mask > 0.5).astype(np.uint8)

    # =====================================
    # CONDITION POSITIVE / NEGATIVE
    # =====================================

    positive_pixels = np.sum(mask)

    THRESHOLD = 500

    # =====================================
    # SI NEGATIVE → RETURN RIEN
    # =====================================

    if positive_pixels <= THRESHOLD:

        return JSONResponse({
            "prediction": "negative"
        })

    # =====================================
    # SI POSITIVE → CONTINUER
    # =====================================

    prediction = "positive"

    # =====================================
    # SAVE MASK IMAGE
    # =====================================

    mask_image = Image.fromarray(mask * 255)

    mask_image = mask_image.resize(original_size)

    mask_filename = f"{base_name}_mask.png"

    mask_path = os.path.join(
        "outputs",
        mask_filename
    )

    mask_image.save(mask_path)

    # =====================================
    # CREATE SEGMENTED IMAGE
    # =====================================

    original_np = np.array(image)

    mask_np = np.array(mask_image)

    overlay = original_np.copy()

    overlay[mask_np > 0] = [255, 0, 0]

    blended = (
        0.6 * original_np +
        0.4 * overlay
    ).astype(np.uint8)

    segmented_image = Image.fromarray(blended)

    segmented_filename = f"{base_name}_segmented{extension}"

    segmented_path = os.path.join(
        "outputs",
        segmented_filename
    )

    segmented_image.save(segmented_path)

    # =====================================
    # RETURN JSON
    # =====================================

    return JSONResponse({

        "prediction": prediction,

        "mask_image":
        f"http://127.0.0.1:8000/outputs/{mask_filename}",

        "segmented_image":
        f"http://127.0.0.1:8000/outputs/{segmented_filename}"
    })
