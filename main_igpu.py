from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image

import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import torch_directml

import numpy as np
import os

# =====================================
# DEVICE iGPU (Intel GPU via DirectML)
# =====================================

device = torch_directml.device()

print("Using device:", device)

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
# LOAD MODEL
# =====================================

# Recréer EXACTEMENT le modèle entraîné
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

# Charger checkpoint
checkpoint = torch.load(
    "model/best_model_resnet101.pth",
    map_location=device
)

# Charger poids
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Envoyer modèle vers iGPU
model.to(device)

# Mode inference
model.eval()

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
        "message": "FastAPI segmentation API is running on iGPU"
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

    # Nom sans extension
    base_name = os.path.splitext(filename)[0]

    # Extension
    extension = os.path.splitext(filename)[1]

    # =====================================
    # LOAD IMAGE
    # =====================================

    image = Image.open(file.file).convert("RGB")

    original_size = image.size

    # =====================================
    # SAVE ORIGINAL IMAGE
    # =====================================

    original_filename = f"{base_name}_original{extension}"

    original_path = os.path.join(
        "outputs",
        original_filename
    )

    image.save(original_path)

    # =====================================
    # PREPROCESSING
    # =====================================

    input_tensor = transform(image).unsqueeze(0).to(device)

    # =====================================
    # PREDICTION
    # =====================================

    with torch.no_grad():

        output = model(input_tensor)

    # =====================================
    # CREATE MASK
    # =====================================

    output = torch.sigmoid(output)

    # Retour CPU avant numpy
    mask = output.squeeze().cpu().numpy()

    # Threshold
    mask = (mask > 0.5).astype(np.uint8)

    # =====================================
    # SAVE MASK IMAGE
    # =====================================

    mask_image = Image.fromarray(mask * 255)

    # Resize taille originale
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

    # Rouge sur zones segmentées
    overlay[mask_np > 0] = [255, 0, 0]

    # Mélange image + overlay
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

        "original_image":
        f"http://127.0.0.1:8000/outputs/{original_filename}",

        "mask_image":
        f"http://127.0.0.1:8000/outputs/{mask_filename}",

        "segmented_image":
        f"http://127.0.0.1:8000/outputs/{segmented_filename}"
    })
