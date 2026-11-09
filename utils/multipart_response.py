import json
import os
import traceback
import uuid
from io import BytesIO
from typing import Callable, Dict, Optional

from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, ImageOps

DATE_SUFFIX = "__date__"


class MultipartBuilder:

    def __init__(self):
        self.boundary = uuid.uuid4().hex
        self._body    = bytearray()

    def _header(self, name: str, content_type: str, filename: Optional[str] = None):
        disposition = f'form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        self._body += (
            f"--{self.boundary}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Disposition: {disposition}\r\n"
            f"\r\n"
        ).encode()

    def add_text(self, name: str, text: str):
        self._header(name, "text/plain")
        self._body += text.encode() + b"\r\n"

    def add_image(self, name: str, filename: str, data: bytes, content_type: str = "image/jpeg"):
        self._header(name, content_type, filename)
        self._body += data + b"\r\n"

    def add_json(self, name: str, payload: dict):
        self._header(name, "application/json")
        self._body += json.dumps(payload, ensure_ascii=False).encode() + b"\r\n"

    def response(self) -> Response:
        self._body += f"--{self.boundary}--\r\n".encode()
        return Response(
            content=bytes(self._body),
            media_type=f"multipart/form-data; boundary={self.boundary}",
        )


def to_float(value) -> Optional[float]:
    """'22,5' ou '22.5' -> 22.5 ; vide / invalide -> None."""
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _is_file(value) -> bool:
    return hasattr(value, "read")


async def process_images(
    request: Request,
    process_fn: Callable,
    empty_label: str,
    extra_fields: Optional[Dict[str, Callable]] = None,
    tag: str = "api",
) -> Response:
    """
    Boucle commune : lit le formulaire, appelle process_fn pour chaque image
    (dans un thread, pour ne pas bloquer la boucle asyncio pendant l'inférence),
    et construit la réponse multipart.

    process_fn(image=PIL RGB, image_bytes, filename, date_client, **extra)
      -> None                         (rien détecté)
      -> (BytesIO jpeg, dict résultat)
    extra_fields : {"nom_du_champ": convertisseur} pour les champs texte en plus.
    """

    form   = await request.form()
    extra_fields = extra_fields or {}

    dates_map: Dict[str, str] = {}
    extra = {name: None for name in extra_fields}

    for key, val in form.multi_items():
        if _is_file(val):
            continue
        if key.endswith(DATE_SUFFIX):
            dates_map[key[: -len(DATE_SUFFIX)]] = str(val)
        elif key in extra_fields:
            extra[key] = extra_fields[key](val)

    builder = MultipartBuilder()

    for image_id, upload in form.multi_items():

        if not _is_file(upload):
            continue

        filename = upload.filename or str(image_id)

        try:
            content = await upload.read()
            try:
                # exif_transpose : les photos de téléphone portrait arrivent dans le bon sens
                image = ImageOps.exif_transpose(Image.open(BytesIO(content))).convert("RGB")
            except Exception:
                print(f"[{tag}] {image_id} ({filename}) : fichier illisible, ignoré")
                continue

            output = await run_in_threadpool(
                process_fn,
                image=image,
                image_bytes=content,
                filename=filename,
                date_client=dates_map.get(filename) or dates_map.get(str(image_id), ""),
                **extra,
            )

            if output is None:
                builder.add_text(image_id, empty_label)
                continue

            image_buffer, result = output
            base_name = os.path.splitext(os.path.basename(filename))[0]

            builder.add_image(image_id, f"{base_name}_segmented.jpg", image_buffer.getvalue())
            builder.add_json(f"{image_id}_criticite", result)

        except Exception as e:
            print(f"[{tag}] erreur sur l'image {image_id} ({filename}) : {e}")
            traceback.print_exc()

        finally:
            await upload.close()

    return builder.response()
