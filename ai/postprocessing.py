import io
import re
import cv2
import numpy as np
import torch

from PIL import Image
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CRACK
# ─────────────────────────────────────────────────────────────────────────────

def create_mask(output):

    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)

    output = torch.sigmoid(output)
    mask   = output.squeeze().numpy()
    mask   = (mask > 0.5).astype(np.uint8)

    return mask


def create_bounding_boxes(image_bgr, zones):

    out = image_bgr.copy()

    for i, z in enumerate(zones):

        x, y, w, h = z["bbox"]
        conf        = z["confidence"]

        if conf >= 0.70:
            color = (0, 255, 100)
        elif conf >= 0.45:
            color = (0, 200, 255)
        else:
            color = (80, 80, 255)

        cv2.rectangle(out, (x, y), (x+w, y+h), color, 1)

        label = str(i + 1)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.35
        thick = 1

        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(out, (x, y), (x + tw + 4, y + th + 4), color, -1)
        cv2.putText(out, label, (x + 2, y + th + 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# HUMIDITY — merge overlapping bboxes (HSV fallback)
# ─────────────────────────────────────────────────────────────────────────────

def merge_overlapping_bboxes(zones, margin=40):
    """
    Fusionne les bounding boxes qui :
    - se chevauchent (overlap direct)
    - sont proches de moins de margin px
    - ont un IoU > 0 (tout overlap compte)
    - ont leurs centres proches (distance < 60% de la diagonale moyenne)

    Répète jusqu'à stabilisation (Union-Find itératif).
    """

    if not zones:
        return zones

    def expand(b, m):
        x, y, w, h = b
        return (x - m, y - m, w + 2*m, h + 2*m)

    def overlaps(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1+w1 < x2 or x2+w2 < x1 or y1+h1 < y2 or y2+h2 < y1)

    def iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        inter = ix * iy
        if inter == 0:
            return 0.0
        return inter / (w1*h1 + w2*h2 - inter)

    def center_dist(b1, b2):
        cx1, cy1 = b1[0] + b1[2]/2, b1[1] + b1[3]/2
        cx2, cy2 = b2[0] + b2[2]/2, b2[1] + b2[3]/2
        return ((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5

    def avg_diag(b1, b2):
        d1 = (b1[2]**2 + b1[3]**2) ** 0.5
        d2 = (b2[2]**2 + b2[3]**2) ** 0.5
        return (d1 + d2) / 2

    def should_merge(b1, b2):
        if overlaps(expand(b1, margin), expand(b2, margin)):
            return True
        if iou(b1, b2) > 0:
            return True
        if center_dist(b1, b2) < avg_diag(b1, b2) * 0.8:
            return True
        return False

    def union_bbox(b1, b2):
        x  = min(b1[0], b2[0])
        y  = min(b1[1], b2[1])
        x2 = max(b1[0]+b1[2], b2[0]+b2[2])
        y2 = max(b1[1]+b1[3], b2[1]+b2[3])
        return (x, y, x2-x, y2-y)

    bboxes   = [z["bbox"]       for z in zones]
    areas    = [z["area"]       for z in zones]
    confs    = [z["confidence"] for z in zones]
    contours = [z["contour"]    for z in zones]

    changed = True

    while changed:

        changed = False
        used    = [False] * len(bboxes)
        nb, na, nc, nct = [], [], [], []

        for i in range(len(bboxes)):

            if used[i]:
                continue

            cb, ca, cc, cct = bboxes[i], areas[i], confs[i], [contours[i]]

            for j in range(i + 1, len(bboxes)):

                if used[j]:
                    continue

                if should_merge(cb, bboxes[j]):
                    new_area = ca + areas[j]
                    cc       = (cc * ca + confs[j] * areas[j]) / new_area
                    ca       = new_area
                    cb       = union_bbox(cb, bboxes[j])
                    cct.append(contours[j])
                    used[j]  = True
                    changed  = True

            nb.append(cb)
            na.append(ca)
            nc.append(round(cc, 3))
            nct.append(np.vstack(cct))

        bboxes, areas, confs, contours = nb, na, nc, nct

    result = [
        {"bbox": b, "area": a, "confidence": c, "contour": ct}
        for b, a, c, ct in zip(bboxes, areas, confs, contours)
    ]

    result.sort(key=lambda z: z["area"], reverse=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HUMIDITY — draw bounding boxes on RGB image  (FLIR Planck pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def draw_humidity_boxes(
    image_rgb: np.ndarray,
    zones: list,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:

    out = image_rgb.copy()

    for z in zones:

        x1 = int(z["x1"] * scale_x)
        y1 = int(z["y1"] * scale_y)
        x2 = int(z["x2"] * scale_x)
        y2 = int(z["y2"] * scale_y)

        r, g, b = z["couleur"]
        bgr     = (b, g, r)

        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 1)

        label = str(z["id"])
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.35
        thick = 1

        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 4, y1 + th + 4), bgr, -1)
        cv2.putText(out, label, (x1 + 2, y1 + th + 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# HUMIDITY — image buffer
# ─────────────────────────────────────────────────────────────────────────────

def image_to_buffer(image_rgb: np.ndarray) -> io.BytesIO:

    buffer = io.BytesIO()

    Image.fromarray(image_rgb).save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True,
    )

    buffer.seek(0)

    return buffer


# ─────────────────────────────────────────────────────────────────────────────
# HUMIDITY — build criticite result  (FLIR Planck pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def build_criticite_result(zones: list, meta: dict, params: dict) -> dict:

    date_str = meta.get("DateTimeOriginal") or meta.get("CreateDate", "")

    date_taken = str(date_str).split("+")[0].strip() if date_str else datetime.now().strftime("%Y:%m:%d %H:%M:%S")

    zones_list = [
        {
            "zone":      z["id"],
            "criticite": z["niveau"],
            "score":     z["score"],
        }
        for z in zones
    ]

    score_moyen   = round(sum(z["score"] for z in zones) / len(zones)) if zones else 0
    ORDRE         = {"CRITIQUE": 3, "MOYEN": 2, "FAIBLE": 1}
    criticite_max = max(zones_list, key=lambda z: ORDRE.get(z["criticite"], 0))["criticite"] if zones_list else "FAIBLE"

    return {
        "date_taken":    date_taken,
        "class":         "humidite",
        "score_moyen":   score_moyen,
        "criticite_max": criticite_max,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HUMIDITY — HSV fallback criticite
# ─────────────────────────────────────────────────────────────────────────────

def build_hsv_criticite_result(zones: list, meta: dict) -> dict:

    date_str = meta.get("DateTimeOriginal") or meta.get("CreateDate", "")

    date_taken = str(date_str).split("+")[0].strip() if date_str else datetime.now().strftime("%Y:%m:%d %H:%M:%S")

    zones_list = []

    for i, z in enumerate(zones):

        score = int(z["confidence"] * 100)

        if   score >= 67: niveau = "CRITIQUE"
        elif score >= 34: niveau = "MOYEN"
        else:             niveau = "FAIBLE"

        zones_list.append({
            "zone":      i + 1,
            "criticite": niveau,
            "score":     score,
        })

    score_moyen   = round(sum(z["score"] for z in zones_list) / len(zones_list)) if zones_list else 0
    ORDRE         = {"CRITIQUE": 3, "MOYEN": 2, "FAIBLE": 1}
    criticite_max = max(zones_list, key=lambda z: ORDRE.get(z["criticite"], 0))["criticite"] if zones_list else "FAIBLE"

    return {
        "date_taken":    date_taken,
        "class":         "humidite",
        "score_moyen":   score_moyen,
        "criticite_max": criticite_max,
    }