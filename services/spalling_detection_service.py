import os

import cv2
import numpy as np

from ai.spalling_model import THRESHOLD, predict_spalling_mask
from utils.date_utils import resolve_date_taken
from utils.image_utils import draw_label, overlay_mask, rgb_to_jpeg_buffer

MIN_ZONE_PCT  = float(os.getenv("SPALLING_MIN_ZONE_PCT", "0.05"))
MORPHO_CLOSE  = int(os.getenv("SPALLING_MORPHO_CLOSE", "5"))
MASK_COLOR    = (255, 0, 255)        # RGB magenta (rouge = fissures)


def _zones(mask, prob):

    h, w = mask.shape
    if MORPHO_CLOSE > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPHO_CLOSE, MORPHO_CLOSE))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    min_area = max(1, int(h * w * MIN_ZONE_PCT / 100.0))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    kept  = np.zeros_like(mask)
    zones = []
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == lbl
        kept[comp] = 1
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        zones.append({
            "area":       area,
            "bbox":       [int(v) for v in stats[lbl, :4]],          # x, y, w, h
            "confidence": float(prob[comp].mean()),
            "contour":    max(contours, key=cv2.contourArea),
        })

    zones.sort(key=lambda z: z["area"], reverse=True)
    return zones, kept


def process_spalling_detection(image, image_bytes=None, filename="", date_client=""):

    image_rgb   = np.array(image)
    mask, prob  = predict_spalling_mask(image_rgb)
    zones, mask = _zones(mask, prob)
    print(f"[spalling] {filename} : {len(zones)} zone(s), {int(mask.sum())} px")

    if not zones:
        return None

    h, w      = mask.shape
    annotated = overlay_mask(image_rgb, mask, color=MASK_COLOR, alpha=0.35)
    thick     = max(1, round(max(h, w) / 600))
    for i, z in enumerate(zones, start=1):
        z["id"] = i
        cv2.drawContours(annotated, [z["contour"]], -1, MASK_COLOR, thick)
        draw_label(annotated, f"{i} {z['confidence']:.2f}", z["bbox"][0], z["bbox"][1], MASK_COLOR)

    result = {
        "date_taken":  resolve_date_taken(date_client, filename, image_bytes),
        "classe":      "spalling",
        "criticite":   None,                                         # détection seule
        "confidence":  round(float(prob[mask.astype(bool)].mean()), 3),
        "nb_zones":    len(zones),
        "surface_pct": round(100.0 * mask.mean(), 2),
        "zones": [
            {
                "zone":        z["id"],
                "confidence":  round(z["confidence"], 3),
                "bbox":        z["bbox"],
                "surface_pct": round(100.0 * z["area"] / (h * w), 2),
            }
            for z in zones
        ],
        "detection":   {"modele": "segformer_mit_b2", "seuil": THRESHOLD},
    }

    return rgb_to_jpeg_buffer(annotated), result
