import cv2
import numpy as np

from PIL import Image

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


def process_humidity_detection(image: Image.Image, image_bytes: bytes = None):

    # ── Pipeline FLIR Planck ──────────────────────────────────────────────────
    if image_bytes is not None:

        meta   = extract_metadata(image_bytes)
        raw    = extract_raw_thermal(image_bytes)

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

    return image_to_buffer(annotated_rgb), criticite_result