import cv2
import numpy as np
import re

from PIL import Image, ExifTags
from io import BytesIO
from datetime import datetime

from ai.preprocessing import preprocess_flir
from ai.inference import predict_humidity_hsv
from ai.postprocessing import (
    merge_overlapping_bboxes,
    draw_humidity_boxes,
    create_bounding_boxes,
    image_to_buffer,
    build_criticite_result,
    build_hsv_criticite_result,
)
from utils.flir_utils import (
    extract_metadata,
    extract_raw_thermal,
    extract_embedded_rgb,
    raw_to_celsius,
    extract_params,
    run_detection_pipeline,
)


def _extract_date(filename: str, image_bytes: bytes) -> str:
    """Même logique que crack_criticality_service."""

    # ── 1. EXIF via Pillow ────────────────────────────────────────────────────
    try:
        img  = Image.open(BytesIO(image_bytes))
        exif = img._getexif()
        if exif:
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            raw  = (
                tags.get("DateTimeOriginal") or
                tags.get("DateTime")         or
                tags.get("DateTimeDigitized") or
                ""
            )
            if raw:
                return str(raw).strip()
    except Exception:
        pass

    # ── 2. Nom du fichier IMG_20260430_094400.jpg ─────────────────────────────
    m = re.search(r"(\d{4})(\d{2})(\d{2})[_\-](\d{2})(\d{2})(\d{2})", filename)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}:{mo}:{d} {h}:{mi}:{s}"

    return ""


def process_humidity_detection(
    image: Image.Image,
    image_bytes: bytes = None,
    filename: str = "",
    date_client: str = "",
):

    # ── Résolution de la date (même priorité que crack) ───────────────────────
    date_taken = (
        date_client.strip()
        or _extract_date(filename, image_bytes or b"")
        or datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    )

    # ── Pipeline FLIR Planck ──────────────────────────────────────────────────
    if image_bytes is not None:

        meta = extract_metadata(image_bytes)
        raw  = extract_raw_thermal(image_bytes)

        if raw is not None:

            temp_c = raw_to_celsius(raw, meta)

            if temp_c is not None:

                params = extract_params(meta)
                zones, infos = run_detection_pipeline(temp_c, meta, params)

                if not zones:
                    return None

                th_h, th_w = temp_c.shape

                rgb = extract_embedded_rgb(
                    image_bytes, shape_thermique=(th_h, th_w)
                )
                if rgb is None:
                    img_arr = np.array(image.convert("RGB"))
                    rgb = cv2.resize(
                        img_arr, (th_w, th_h), interpolation=cv2.INTER_LANCZOS4
                    )

                annotated        = draw_humidity_boxes(rgb, zones, scale_x=1.0, scale_y=1.0)
                criticite_result = build_criticite_result(zones, meta, params)

                # ── Format identique à crack ──────────────────────────────────
                criticite_result["date_taken"] = date_taken
                criticite_result["classe"]     = "humidite"

                return image_to_buffer(annotated), criticite_result

    # ── Fallback HSV ──────────────────────────────────────────────────────────
    meta = {}

    image_bgr, hsv = preprocess_flir(image)
    zones, mask    = predict_humidity_hsv(hsv)
    zones          = merge_overlapping_bboxes(zones, margin=40)

    if not zones:
        return None

    annotated_bgr    = create_bounding_boxes(image_bgr, zones)
    annotated_rgb    = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    criticite_result = build_hsv_criticite_result(zones, meta)

    # ── date_taken depuis priorité client > EXIF > fichier > now ───────────────
    criticite_result["date_taken"] = date_taken

    return image_to_buffer(annotated_rgb), criticite_result