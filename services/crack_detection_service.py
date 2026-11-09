import os

import numpy as np

from ai.crack_model import predict_crack_mask
from services.crack_criticality_service import process_criticite
from utils.image_utils import overlay_mask, rgb_to_jpeg_buffer

MIN_CRACK_PIXELS = int(os.getenv("CRACK_MIN_PIXELS", "50"))


def process_crack_detection(image, image_bytes=None, filename="", date_client=""):

    image_rgb    = np.array(image)
    pred_bin     = predict_crack_mask(image_rgb)
    crack_pixels = int(pred_bin.sum())
    print(f"[crack] {filename} : {crack_pixels} px de fissure")

    if crack_pixels < MIN_CRACK_PIXELS:
        return None

    criticite_result = process_criticite(
        image_rgb,
        image_bytes=image_bytes,
        filename=filename,
        date_client=date_client,
    )

    overlay = overlay_mask(image_rgb, pred_bin, color=(255, 0, 0), alpha=0.25)
    return rgb_to_jpeg_buffer(overlay, quality=75), criticite_result
