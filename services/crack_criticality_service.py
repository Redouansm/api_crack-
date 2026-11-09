import numpy as np
from PIL import Image

from ai.crack_criticality_model import predict_criticite
from utils.date_utils import resolve_date_taken


def process_criticite(image_rgb, image_bytes=None, filename="", date_client=""):

    image_pil = image_rgb if isinstance(image_rgb, Image.Image) else Image.fromarray(np.asarray(image_rgb))
    result    = predict_criticite(image_pil)

    return {
        "date_taken": resolve_date_taken(date_client, filename, image_bytes),
        "classe":     "crack",
        "criticite":  result["criticite"],
        "confidence": result["confidence"],
    }
