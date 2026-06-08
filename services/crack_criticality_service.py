from PIL import Image, ExifTags
from datetime import datetime
from io import BytesIO
import re

from ai.criticite_inference import predict_criticite


def _extract_date(filename: str, image_bytes: bytes) -> str:

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

    # ── 2. Nom du fichier  IMG_20260430_094400.jpg ────────────────────────────
    m = re.search(r"(\d{4})(\d{2})(\d{2})[_\-](\d{2})(\d{2})(\d{2})", filename)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}:{mo}:{d} {h}:{mi}:{s}"

    return ""


def process_criticite(image_rgb, image_bytes=None, filename="", date_client=""):

    if not isinstance(image_rgb, Image.Image):
        image_pil = Image.fromarray(image_rgb)
    else:
        image_pil = image_rgb

    result = predict_criticite(image=image_pil, n_aug=10)

    # Priorité : client > EXIF > nom fichier > now
    date_taken = (
        date_client.strip()
        or _extract_date(filename, image_bytes or b"")
        or datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    )

    return {
        "date_taken": date_taken,
        "classe":     "crack",
        "criticite":  result["Criticité"],
        "confidence": result["confidence"],
    }