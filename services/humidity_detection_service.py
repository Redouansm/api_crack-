import cv2
import numpy as np

from PIL import Image

from io import BytesIO

from ai.preprocessing import (
    preprocess_image
)

from ai.inference import (
    predict_humidity
)

from ai.postprocessing import (
    create_bounding_boxes
)


def process_humidity_detection(
    image
):

    image_bgr, hsv = preprocess_image(
        image
    )

    zones, mask = predict_humidity(
        hsv
    )

    # =====================================
    # NO HUMIDITY
    # =====================================

    if len(zones) == 0:

        return None

    # =====================================
    # DRAW BOXES
    # =====================================

    segmented_image = (
        create_bounding_boxes(
            image_bgr,
            zones
        )
    )

    segmented_image = Image.fromarray(

        cv2.cvtColor(

            segmented_image,

            cv2.COLOR_BGR2RGB
        )
    )

    # =====================================
    # JPEG BUFFER
    # =====================================

    buffer = BytesIO()

    segmented_image.save(

        buffer,

        format="JPEG",

        quality=70,

        optimize=True
    )

    buffer.seek(0)

    return buffer