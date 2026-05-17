import torch
import numpy as np

def create_mask(output):

    if isinstance(
        output,
        np.ndarray
    ):

        output = torch.from_numpy(
            output
        )

    output = torch.sigmoid(output)

    mask = output.squeeze().numpy()

    mask = (
        mask > 0.5
    ).astype(np.uint8)

    return mask



#------------------------------------------------------------------------------#
#                                   HUMIDITY                                   #          
#------------------------------------------------------------------------------#

import cv2

def create_bounding_boxes(
    image_bgr,
    zones
):

    out = image_bgr.copy()

    for i, z in enumerate(zones):

        x, y, w, h = z["bbox"]

        conf = z["confidence"]

        if conf >= 0.70:

            color = (0, 255, 100)

        elif conf >= 0.45:

            color = (0, 200, 255)

        else:

            color = (80, 80, 255)

        cv2.rectangle(

            out,

            (x, y),

            (x+w, y+h),

            color,

            3
        )

        overlay = out.copy()

        cv2.drawContours(

            overlay,

            [z["contour"]],

            -1,

            color,

            thickness=-1
        )

        cv2.addWeighted(

            overlay,

            0.20,

            out,

            0.80,

            0,

            out
        )

        label = (
            f"Humidite #{i+1} "
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = max(
            0.45,
            min(0.85, w / 250)
        )

        thick = 2

        (tw, th), _ = cv2.getTextSize(
            label,
            font,
            scale,
            thick
        )

        ty = y + th + 6

        cv2.rectangle(

            out,

            (x, y),

            (x + tw + 8, y + th + 10),

            color,

            -1
        )

        cv2.putText(

            out,

            label,

            (x + 4, ty),

            font,

            scale,

            (0, 0, 0),

            thick,

            cv2.LINE_AA
        )

    return out