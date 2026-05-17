from fastapi import APIRouter
from fastapi import Request

from fastapi.responses import (
    Response
)

from PIL import Image

from io import BytesIO

import uuid
import os

from services.humidity_detection_service import (
    process_humidity_detection
)

# =====================================
# ROUTER
# =====================================

router = APIRouter(

    tags=["Humidity Detection"]
)


# =====================================
# PREDICT
# =====================================

@router.post("/predict")
async def predict_humidity(request: Request):

    # =====================================
    # RECEIVE FORM DATA
    # =====================================

    form = await request.form()

    # =====================================
    # MULTIPART BOUNDARY
    # =====================================

    boundary = uuid.uuid4().hex

    body = b""

    # =====================================
    # LOOP THROUGH IMAGES
    # =====================================

    for image_id, upload_file in form.items():

        try:

            # =====================================
            # READ IMAGE
            # =====================================

            content = await upload_file.read()

            image = Image.open(
                BytesIO(content)
            ).convert("RGB")

            # =====================================
            # PROCESS DETECTION
            # =====================================

            result = process_humidity_detection(
                image=image
            )

            # =====================================
            # NO HUMIDITY
            # RETURN ONLY IMAGE ID
            # =====================================

            if result is None:

                body += (
                    f"--{boundary}\r\n"
                    f"Content-Type: text/plain\r\n"
                    f'Content-Disposition: form-data; '
                    f'name="{image_id}"\r\n'
                    f"\r\n"
                    f"{image_id}\r\n"
                ).encode()

                continue

            # =====================================
            # HUMIDITY DETECTED
            # RETURN IMAGE + ID
            # =====================================

            original_name = (
                upload_file.filename
            )

            base_name = os.path.splitext(
                original_name
            )[0]

            filename = (
                f"{base_name}_segmented.jpg"
            )

            body += (
                f"--{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f'Content-Disposition: form-data; '
                f'name="{image_id}"; '
                f'filename="{filename}"\r\n'
                f"\r\n"
            ).encode()

            body += result.getvalue()

            body += b"\r\n"

        except Exception as e:

            print(
                f"Error processing image "
                f"{image_id}: {e}"
            )

            continue

        finally:

            await upload_file.close()

    # =====================================
    # CLOSE MULTIPART
    # =====================================

    body += (
        f"--{boundary}--\r\n"
    ).encode()

    # =====================================
    # RETURN RESPONSE
    # =====================================

    return Response(

        content=body,

        media_type=(
            f"multipart/mixed; "
            f"boundary={boundary}"
        )
    )