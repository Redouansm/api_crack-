from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import Response

from PIL import Image

from io import BytesIO

import uuid
import os
import json

from services.crack_detection_service import (
    process_crack_detection
)

router = APIRouter()


@router.post("/predict")
async def predict_crack(request: Request):

    form = await request.form()

    # ── Collecter les dates envoyées par le frontend ──────────────────────────
    dates_map = {}
    for key, val in form.items():
        if key.endswith("__date__") and not hasattr(val, "read"):
            image_name = key.replace("__date__", "")
            dates_map[image_name] = str(val)
    print(f"[crack_route] dates_map = {dates_map}")
    print(f"[crack_route] form keys = {list(form.keys())}")

    boundary = uuid.uuid4().hex
    body     = b""

    for image_id, upload_file in form.items():

        # Ignorer les champs texte
        if not hasattr(upload_file, "read"):
            continue

        try:

            content = await upload_file.read()
            image   = Image.open(BytesIO(content)).convert("RGB")

            # Date depuis le frontend ou depuis les EXIF
            date_from_client = dates_map.get(upload_file.filename or image_id, "")

            response = process_crack_detection(
                image=image,
                image_bytes=content,
                filename=upload_file.filename or "",
                date_client=date_from_client,
            )

            if response is None:
                body += (
                    f"--{boundary}\r\n"
                    f"Content-Type: text/plain\r\n"
                    f'Content-Disposition: form-data; name="{image_id}"\r\n'
                    f"\r\n"
                    f"no_crack\r\n"
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

            raw = image_buffer.getvalue()
            print(f"first bytes hex: {raw[:10].hex()}")
            body += raw
            body += b"\r\n"

            criticite_json = json.dumps(criticite_result, ensure_ascii=False)

            body += (
                f"--{boundary}\r\n"
                f"Content-Type: application/json\r\n"
                f'Content-Disposition: form-data; name="{image_id}_criticite"\r\n'
                f"\r\n"
                f"{criticite_json}\r\n"
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