import os

import cv2
import numpy as np

from ai.device import LazyModel, model_path, torch_device

UPSCALE = int(os.getenv("HUMIDITY_UPSCALE", "4"))       # 160x120 -> 640x480

BLUE_LOWER = np.array([100, 40, 30])
BLUE_UPPER = np.array([160, 255, 255])

MIN_TOUCH_RATIO        = float(os.getenv("HUMIDITY_MIN_TOUCH_RATIO", "0.10"))
MAX_SEGMENT_AREA_RATIO = 0.6
MIN_ZONE_AREA_PX       = int(200 * (UPSCALE / 4) ** 2)  # 200 px à x4

SAM_CONF   = float(os.getenv("HUMIDITY_SAM_CONF", "0.05"))
SAM_IMGSZ  = int(os.getenv("HUMIDITY_SAM_IMGSZ", "640"))

_SAM_WEIGHTS = os.getenv("HUMIDITY_SAM_WEIGHTS", model_path("humidity", "FastSAM-s.pt"))

def _load_sam():
    from ultralytics import FastSAM

    weights = _SAM_WEIGHTS if os.path.exists(_SAM_WEIGHTS) else "FastSAM-s.pt"
    device  = 0 if torch_device() == "cuda" else "cpu"
    print(f"[humidity] FastSAM ({weights}) sur {'GPU' if device == 0 else 'CPU'}")
    return {"model": FastSAM(weights), "device": device}


SAM_MODEL = LazyModel("humidity_fastsam", _load_sam)


def get_sam():

    if SAM_MODEL.status.startswith("error"):
        return None
    try:
        return SAM_MODEL.get()
    except Exception as e:
        print(f" FastSAM indisponible ({e}) -> masque bleu seul")
        return None


def _segments_sam(gray_big: np.ndarray) -> list:
    sam = get_sam()
    if sam is None:
        return []

    rgb = cv2.cvtColor(gray_big, cv2.COLOR_GRAY2RGB)
    h, w = gray_big.shape

    try:
        with SAM_MODEL.lock:
            results = sam["model"](
                rgb, device=sam["device"], retina_masks=True,
                imgsz=SAM_IMGSZ, conf=SAM_CONF, iou=0.9, verbose=False,
            )
    except Exception as e:
        print(f" FastSAM a échoué ({e}) -> masque bleu seul")
        return []

    if results[0].masks is None:
        return []

    masks = results[0].masks.data.cpu().numpy()
    return [
        cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        for m in masks
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de détection
# ─────────────────────────────────────────────────────────────────────────────

def detecter_zones_froides(temp_c: np.ndarray) -> dict:


    h, w = temp_c.shape
    W, H = w * UPSCALE, h * UPSCALE

    tmin, tmax = float(np.nanmin(temp_c)), float(np.nanmax(temp_c))
    norm  = np.clip((temp_c - tmin) / max(tmax - tmin, 0.01), 0, 1)
    gray8 = np.nan_to_num(norm * 255).astype(np.uint8)

    gray_big   = cv2.resize(gray8, (W, H), interpolation=cv2.INTER_CUBIC)
    temp_big   = cv2.resize(temp_c.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    colored    = cv2.applyColorMap(gray_big, cv2.COLORMAP_JET)          # BGR
    hsv        = cv2.cvtColor(colored, cv2.COLOR_BGR2HSV)
    blue_mask  = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    # ── SAM ∩ bleu ───────────────────────────────────────────────────────────
    segments  = _segments_sam(gray_big)
    img_area  = blue_mask.size
    area_min  = max(int(img_area * 0.0005), 1)
    area_max  = int(img_area * MAX_SEGMENT_AREA_RATIO)

    final_mask = np.zeros_like(blue_mask)
    sam_used   = False
    blue_bool  = blue_mask > 0

    for seg in segments:
        seg_bool = seg > 0
        area     = int(seg_bool.sum())
        if area < area_min or area > area_max:
            continue
        inter = seg_bool & blue_bool
        if inter.sum() / area < MIN_TOUCH_RATIO:
            continue
        final_mask[inter] = 255
        sam_used = True

    if not sam_used:
        final_mask = blue_mask.copy()

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN,  kernel)

    # ── Composantes connexes -> zones ────────────────────────────────────────
    n, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    zones = []

    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < MIN_ZONE_AREA_PX:
            continue
        comp = (labels == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        zones.append({
            "label":   lbl,
            "mask":    comp.astype(bool),
            "area":    area,
            "bbox":    tuple(int(v) for v in stats[lbl, :4]),   # x, y, w, h
            "contour": max(contours, key=cv2.contourArea),
        })

    zones.sort(key=lambda z: z["area"], reverse=True)

    return {
        "temp_big":     temp_big,
        "colored_bgr":  colored,
        "final_mask":   final_mask > 0,
        "zones":        zones,
        "sam_used":     sam_used,
        "nb_segments":  len(segments),
    }


# ─────────────────────────────────────────────────────────────────────────────

def predict_humidity_hsv(image_rgb: np.ndarray, min_area=None):

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv       = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    img_h, img_w = hsv.shape[:2]
    img_area     = img_h * img_w

    m1   = cv2.inRange(hsv, np.array([100, 60, 10]), np.array([165, 255, 160]))
    m2   = cv2.inRange(hsv, np.array([0, 150, 10]),  np.array([30, 255, 100]))
    mask = cv2.bitwise_or(m1, m2)

    # Exclure l'interface FLIR (température/logo en haut, palette à gauche, bords)
    mask[:int(img_h * 0.12), :] = 0
    mask[int(img_h * 0.92):, :] = 0
    mask[:, :int(img_w * 0.15)] = 0
    mask[:, int(img_w * 0.90):] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min    = max(min_area or 0, int(img_area * 0.01))

    zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        density    = np.sum(mask[y:y + h, x:x + w] > 0) / max(w * h, 1)
        zones.append({
            "bbox":       (x, y, w, h),
            "area":       int(area),
            "confidence": round(float(density), 3),
            "contour":    cnt,
        })

    zones.sort(key=lambda z: z["area"], reverse=True)

    # Toute zone < 25 % de la plus grande est du bruit
    if len(zones) > 1:
        zones = [z for z in zones if z["area"] >= zones[0]["area"] * 0.25]

    return zones, image_bgr


# ─────────────────────────────────────────────────────────────────────────────
# REPLI HSV — fusion des boîtes qui se chevauchent / sont proches
# ─────────────────────────────────────────────────────────────────────────────

def merge_overlapping_bboxes(zones, margin=40):

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
