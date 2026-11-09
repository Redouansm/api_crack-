# ─────────────────────────────────────────────────────────────────────────────
# CHUTES DE ZELLIGE — détection seule (pas encore de modèle de criticité)
# Le JSON garde la même forme que crack/humidité ; "criticite" vaut null.
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np

from ai.chute_model import CONF, IMGSZ, predict_chutes
from utils.date_utils import resolve_date_taken
from utils.image_utils import draw_label, rgb_to_jpeg_buffer

BOX_COLOR       = (255, 140, 0)      # RGB orange
MAX_LABELS      = 60                 # au-delà, boîtes sans étiquette (lisibilité)


def _dessiner(image_rgb, detections):
    out   = image_rgb.copy()
    thick = max(1, round(max(out.shape[:2]) / 600))

    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thick)

    if len(detections) <= MAX_LABELS:
        for d in detections:
            x1, y1, _, _ = d["bbox_xyxy"]
            draw_label(out, f"{d['id']} {d['confidence']:.2f}", x1, y1, BOX_COLOR)

    return out


def process_chute_detection(image, image_bytes=None, filename="", date_client=""):

    image_rgb  = np.array(image)
    detections = predict_chutes(image_rgb)
    print(f"[chute] {filename} : {len(detections)} élément(s) manquant(s)")

    if not detections:
        return None

    h, w     = image_rgb.shape[:2]
    covered  = np.zeros((h, w), dtype=bool)

    zones = []
    for i, d in enumerate(detections, start=1):
        d["id"] = i
        x1, y1, x2, y2 = d["bbox_xyxy"]
        covered[max(y1, 0):y2, max(x1, 0):x2] = True
        zones.append({
            "zone":        i,
            "classe":      d["classe"],
            "confidence":  round(d["confidence"], 3),
            "bbox":        [x1, y1, x2 - x1, y2 - y1],          # x, y, w, h (comme humidité)
            "surface_pct": round(100.0 * (x2 - x1) * (y2 - y1) / (w * h), 2),
        })

    result = {
        "date_taken":    resolve_date_taken(date_client, filename, image_bytes),
        "classe":        "chute",
        "criticite":     None,                                   # détection seule
        "confidence":    round(max(d["confidence"] for d in detections), 3),
        "nb_elements":   len(detections),
        "surface_pct":   round(100.0 * covered.mean(), 2),       # union des boîtes
        "zones":         zones,
        "detection":     {"modele": "yolov12", "imgsz": IMGSZ, "conf_min": CONF},
    }

    return rgb_to_jpeg_buffer(_dessiner(image_rgb, detections)), result
