from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import Response

from PIL import Image

from io import BytesIO

import uuid
import os

from services.crack_detection_service import (
    process_crack_detection
)

router = APIRouter()


@router.post("/predict")
async def predict_crack(request: Request):

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

            content = await upload_file.read()

            image = Image.open(
                BytesIO(content)
            ).convert("RGB")

            result = process_crack_detection(
                image=image
            )

            # =====================================
            # NO CRACK
            # =====================================

            if result is None:

                body += (
                    f"--{boundary}\r\n"
                    f"Content-Type: text/plain\r\n"
                    f'Content-Disposition: form-data; '
                    f'name="{image_id}"\r\n'
                    f"\r\n"
                    f"no_crack\r\n"
                ).encode()

                continue

            # =====================================
            # CRACK DETECTED
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

            raw = result.getvalue()
            print(f"first bytes hex: {raw[:10].hex()}")
            body += raw

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

    return Response(

        content=body,

        media_type=(
            f"multipart/form-data; "
            f"boundary={boundary}"
        )
    )