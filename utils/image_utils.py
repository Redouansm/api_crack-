import base64
import io

import cv2
import numpy as np
from PIL import Image

JPEG_QUALITY = 85


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def rgb_to_jpeg_buffer(image_rgb: np.ndarray, quality: int = JPEG_QUALITY) -> io.BytesIO:
    """ndarray RGB uint8 -> BytesIO JPEG (prêt pour la réponse multipart)."""
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise ValueError("Encodage JPEG impossible")
    return io.BytesIO(encoded.tobytes())


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color=(255, 0, 0), alpha: float = 0.25) -> np.ndarray:
    """Teinte les pixels du masque avec `color` (RGB) en transparence `alpha`."""
    out = image_rgb.astype(np.float32)
    sel = mask.astype(bool)
    out[sel] = out[sel] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def draw_label(image_rgb: np.ndarray, text: str, x: int, y: int, color, scale: float = None) -> None:
    """Étiquette pleine au-dessus du point (x, y), texte noir. Modifie l'image en place."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = scale or max(0.4, image_rgb.shape[1] / 1400)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    ty = y if y - th - 6 < 0 else y - th - 6
    cv2.rectangle(image_rgb, (x, ty), (x + tw + 6, ty + th + 6), color, -1)
    cv2.putText(image_rgb, text, (x + 3, ty + th + 2), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
