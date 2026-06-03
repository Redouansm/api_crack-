from fastapi import APIRouter, Request
from fastapi.responses import Response

from PIL import Image
from io import BytesIO

import uuid
import os
import json

from services.humidity_detection_service import process_humidity_detection

router = APIRouter()


@router.post("/predict")
async def predict_humidity(request: Request):

    form     = await request.form()
    boundary = uuid.uuid4().hex
    body     = b""

    for image_id, upload_file in form.items():

        if not hasattr(upload_file, "read"):
            continue

        try:

            content = await upload_file.read()
            image   = Image.open(BytesIO(content)).convert("RGB")

            response = process_humidity_detection(
                image=image,
                image_bytes=content,
            )

            if response is None:
                body += (
                    f"--{boundary}\r\n"
                    f"Content-Type: text/plain\r\n"
                    f'Content-Disposition: form-data; name="{image_id}"\r\n'
                    f"\r\n"
                    f"no_humidity\r\n"
                ).encode()
                continue

            image_buffer, criticite_result = response

            base_name = os.path.splitext(upload_file.filename)[0]
            filename  = f"{base_name}_segmented.jpg"

            body += (
                f"--{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f'Content-Disposition: form-data; name="{image_id}"; filename="{filename}"\r\n'
                f"\r\n"
            ).encode()
            body += image_buffer.getvalue()
            body += b"\r\n"

            body += (
                f"--{boundary}\r\n"
                f"Content-Type: application/json\r\n"
                f'Content-Disposition: form-data; name="{image_id}_criticite"\r\n'
                f"\r\n"
                f"{json.dumps(criticite_result, ensure_ascii=False)}\r\n"
            ).encode()

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue

        finally:
            await upload_file.close()

    body += f"--{boundary}--\r\n".encode()

    return Response(
        content=body,
        media_type=f"multipart/form-data; boundary={boundary}"
    )