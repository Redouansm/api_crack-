import re
from datetime import datetime
from io import BytesIO

from PIL import ExifTags, Image

DATE_FORMAT = "%Y:%m:%d %H:%M:%S"

_FILENAME_DATE = re.compile(r"(\d{4})(\d{2})(\d{2})[_\-](\d{2})(\d{2})(\d{2})")


def extract_date(filename: str, image_bytes: bytes) -> str:

    try:
        exif = Image.open(BytesIO(image_bytes)).getexif()
        merged = {**exif, **exif.get_ifd(ExifTags.IFD.Exif)}
        tags   = {ExifTags.TAGS.get(k, k): v for k, v in merged.items()}
        raw    = tags.get("DateTimeOriginal") or tags.get("DateTime") or tags.get("DateTimeDigitized") or ""
        if raw:
            return str(raw).strip()
    except Exception:
        pass

    m = _FILENAME_DATE.search(filename or "")
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}:{mo}:{d} {h}:{mi}:{s}"

    return ""


def resolve_date_taken(date_client: str = "", filename: str = "", image_bytes: bytes = None) -> str:
    return (
        (date_client or "").strip()
        or extract_date(filename, image_bytes or b"")
        or datetime.now().strftime(DATE_FORMAT)
    )
