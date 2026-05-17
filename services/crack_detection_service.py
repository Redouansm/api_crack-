import cv2
import torch
import numpy as np

from io import BytesIO

from ai.model_loader import (
    load_model
)

# =====================================
# CONFIG
# =====================================

INPUT_SIZE = 512

THRESHOLD = 0.95

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

# =====================================
# LOAD MODEL
# =====================================

MODEL_DATA = load_model(
    "crack"
)

BACKEND = MODEL_DATA["backend"]

MODEL = MODEL_DATA["model"]

print(
    "Backend:",
    BACKEND
)

# =====================================
# PREPROCESS
# =====================================

def preprocess_image(image_rgb):

    original_h, original_w = (
        image_rgb.shape[:2]
    )

    # =====================================
    # KEEP RATIO
    # =====================================

    scale = INPUT_SIZE / max(
        original_h,
        original_w
    )

    new_w = int(
        original_w * scale
    )

    new_h = int(
        original_h * scale
    )

    resized = cv2.resize(

        image_rgb,

        (
            new_w,
            new_h
        ),

        interpolation=cv2.INTER_AREA
    )

    # =====================================
    # PAD
    # =====================================

    padded = np.zeros(

        (
            INPUT_SIZE,
            INPUT_SIZE,
            3
        ),

        dtype=np.uint8
    )

    padded[
        :new_h,
        :new_w
    ] = resized

    # =====================================
    # NORMALIZE
    # =====================================

    x = (
        padded.astype(np.float32)
        / 255.0
    )

    x = (
        x - MEAN
    ) / STD

    x = np.transpose(
        x,
        (2, 0, 1)
    )

    x = np.expand_dims(
        x,
        axis=0
    )

    return (
        x,
        original_h,
        original_w,
        new_h,
        new_w
    )

# =====================================
# PREDICT
# =====================================

def predict_full_image(image_rgb):

    (
        x,
        original_h,
        original_w,
        new_h,
        new_w
    ) = preprocess_image(
        image_rgb
    )

    # =====================================
    # OPENVINO
    # =====================================

    if BACKEND == "openvino":

        output = MODEL(
            [x]
        )

        pred = list(
            output.values()
        )[0]

        pred = pred.squeeze()

        pred = (
            1 / (
                1 + np.exp(-pred)
            )
        )

    # =====================================
    # CUDA
    # =====================================

    elif BACKEND == "cuda":

        tensor = torch.from_numpy(
            x
        ).half().to("cuda")

        with torch.inference_mode():

            pred = MODEL(
                tensor
            )

            pred = torch.sigmoid(
                pred
            )

            pred = (
                pred.squeeze()
                .float()
                .cpu()
                .numpy()
            )

    # =====================================
    # CPU
    # =====================================

    else:

        tensor = torch.from_numpy(
            x
        ).float()

        with torch.inference_mode():

            pred = MODEL(
                tensor
            )

            pred = torch.sigmoid(
                pred
            )

            pred = (
                pred.squeeze()
                .cpu()
                .numpy()
            )

    # =====================================
    # REMOVE PADDING
    # =====================================

    pred = pred[
        :new_h,
        :new_w
    ]

    # =====================================
    # THRESHOLD
    # =====================================

    pred_bin = (
        pred > THRESHOLD
    ).astype(np.uint8)

    # =====================================
    # RESIZE BACK
    # =====================================

    pred_bin = cv2.resize(

        pred_bin,

        (
            original_w,
            original_h
        ),

        interpolation=cv2.INTER_NEAREST
    )

    return pred_bin

# =====================================
# MAIN
# =====================================

def process_crack_detection(image):

    image_rgb = np.array(
        image
    )

    pred_bin = predict_full_image(
        image_rgb
    )

    crack_pixels = pred_bin.sum()

    print(
        "crack_pixels:",
        crack_pixels
    )

    if crack_pixels < 50:

        return None

    # =====================================
    # OVERLAY
    # =====================================

    overlay = (
        image_rgb.astype(np.float32)
        / 255.0
    )

    red_mask = (
        pred_bin == 1
    )

    alpha = 0.25

    overlay[
        red_mask
    ] = (

        overlay[
            red_mask
        ] * (1 - alpha)

        +

        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        ) * alpha
    )

    overlay = (
        overlay * 255
    ).astype(np.uint8)

    # =====================================
    # JPEG
    # =====================================

    overlay_bgr = cv2.cvtColor(

        overlay,

        cv2.COLOR_RGB2BGR
    )

    _, encoded = cv2.imencode(

        ".jpg",

        overlay_bgr,

        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            75
        ]
    )

    return BytesIO(
        encoded.tobytes()
    )