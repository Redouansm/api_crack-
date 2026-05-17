import torch

from ai.model_loader import load_model

def predict(
    input_tensor,
    model_name
):

    loaded = load_model(model_name)

    backend = loaded["backend"]

    model = loaded["model"]

    # =====================================
    # CUDA
    # =====================================

    if backend == "cuda":

        input_tensor = torch.tensor(
            input_tensor
        ).cuda()

        with torch.no_grad():

            output = model(input_tensor)

        return output.cpu().numpy()

    # =====================================
    # OPENVINO
    # =====================================

    elif backend == "openvino":

        try:

            infer_request = (
                model.create_infer_request()
            )

            results = infer_request.infer({
                0: input_tensor
            })

            output = results[
                model.output(0)
            ]

            return output

        except Exception as e:

            print(
                "OpenVINO failed:",
                e
            )

            print(
                "Fallback to CPU"
            )

            pytorch_model = loaded[
                "pytorch_model"
            ]

            input_tensor = torch.tensor(
                input_tensor
            )

            with torch.no_grad():

                output = pytorch_model(
                    input_tensor
                )

            return output.numpy()

    # =====================================
    # CPU
    # =====================================

    else:

        input_tensor = torch.tensor(
            input_tensor
        )

        with torch.no_grad():

            output = model(input_tensor)

        return output.numpy()
    



#------------------------------------------------------------------------------#
#                                   HUMIDITY                                   #          
#------------------------------------------------------------------------------#
import cv2
import numpy as np

PALETTE_PROFILES = {
    "iron": [
        {
            "lower": np.array([140, 80, 20]),
            "upper": np.array([179, 255, 255])
        },

        {
            "lower": np.array([100, 80, 20]),
            "upper": np.array([140, 255, 200])
        },
    ],

    "rainbow": [
        {
            "lower": np.array([85, 80, 20]),
            "upper": np.array([140, 255, 230])
        },
    ],

    "auto": [
        {
            "lower": np.array([85, 80, 20]),
            "upper": np.array([179, 255, 255])
        },
    ],
}

SENSITIVITY_CONFIGS = {

    "low": {
        "min_area": 2000,
        "kernel": 9,
        "iterations": 3
    },

    "medium": {
        "min_area": 500,
        "kernel": 9,
        "iterations": 3
    },

    "high": {
        "min_area": 200,
        "kernel": 5,
        "iterations": 2
    },
}

UI_EXCLUSION = {
    "x_max": 200,
    "y_max": 160
}

def is_ui_zone(cnt):

    x, y, w, h = cv2.boundingRect(cnt)

    cx = x + w // 2
    cy = y + h // 2

    return (
        cx < UI_EXCLUSION["x_max"]
        and
        cy < UI_EXCLUSION["y_max"]
    )

def merge_overlapping_bboxes(
    zones,
    margin=2
):

    if not zones:
        return zones

    def expand(b, m):

        return (
            b[0]-m,
            b[1]-m,
            b[2]+2*m,
            b[3]+2*m
        )

    def overlaps(b1, b2):

        x1,y1,w1,h1 = b1
        x2,y2,w2,h2 = b2

        return not (

            x1+w1 < x2
            or
            x2+w2 < x1
            or
            y1+h1 < y2
            or
            y2+h2 < y1
        )

    def union(b1, b2):

        x = min(b1[0], b2[0])
        y = min(b1[1], b2[1])

        x2 = max(
            b1[0]+b1[2],
            b2[0]+b2[2]
        )

        y2 = max(
            b1[1]+b1[3],
            b2[1]+b2[3]
        )

        return (
            x,
            y,
            x2-x,
            y2-y
        )

    bboxes = [z["bbox"] for z in zones]
    areas = [z["area"] for z in zones]
    confs = [z["confidence"] for z in zones]
    contours = [z["contour"] for z in zones]

    changed = True

    while changed:

        changed = False

        used = [False] * len(bboxes)

        nb, na, nc, nct = [], [], [], []

        for i in range(len(bboxes)):

            if used[i]:
                continue

            cb, ca, cc, cct = (
                bboxes[i],
                areas[i],
                confs[i],
                [contours[i]]
            )

            for j in range(i+1, len(bboxes)):

                if used[j]:
                    continue

                if overlaps(
                    expand(cb, margin),
                    expand(bboxes[j], margin)
                ):

                    new_area = ca + areas[j]

                    cc = (
                        cc * ca
                        +
                        confs[j] * areas[j]
                    ) / new_area

                    ca = new_area

                    cb = union(
                        cb,
                        bboxes[j]
                    )

                    cct.append(contours[j])

                    used[j] = True

                    changed = True

            nb.append(cb)
            na.append(ca)
            nc.append(round(cc, 3))
            nct.append(np.vstack(cct))

        bboxes, areas, confs, contours = (
            nb,
            na,
            nc,
            nct
        )

    result = [

        {
            "bbox": b,
            "area": a,
            "confidence": c,
            "contour": ct
        }

        for b, a, c, ct in zip(
            bboxes,
            areas,
            confs,
            contours
        )
    ]

    result.sort(
        key=lambda z: z["area"],
        reverse=True
    )

    return result

def predict_humidity(
    hsv,
    palette="iron",
    sensitivity="medium",
    min_area=None,
    merge_margin=5
):

    cfg = SENSITIVITY_CONFIGS[sensitivity]

    k = cfg["kernel"]

    iters = cfg["iterations"]

    area_thresh = (
        min_area
        if min_area is not None
        else cfg["min_area"]
    )

    profiles = PALETTE_PROFILES.get(
        palette,
        PALETTE_PROFILES["iron"]
    )

    mask = np.zeros(
        hsv.shape[:2],
        dtype=np.uint8
    )

    for p in profiles:

        m = cv2.inRange(
            hsv,
            p["lower"],
            p["upper"]
        )

        mask = cv2.bitwise_or(
            mask,
            m
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iters
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    zones = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < area_thresh:
            continue

        if is_ui_zone(cnt):
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        roi_mask = mask[y:y+h, x:x+w]

        density = (
            np.sum(roi_mask > 0)
            /
            (w * h)
        )

        confidence = round(
            float(density),
            3
        )

        zones.append({

            "bbox": (x,y,w,h),

            "area": int(area),

            "confidence": confidence,

            "contour": cnt
        })

    zones.sort(
        key=lambda z: z["area"],
        reverse=True
    )

    zones = merge_overlapping_bboxes(
        zones,
        margin=merge_margin
    )

    return zones, mask