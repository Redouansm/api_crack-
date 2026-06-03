import cv2
import numpy as np

from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# CRACK
# ─────────────────────────────────────────────────────────────────────────────

def resize_image(image, size=448):

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.numpy()
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = np.ascontiguousarray(input_tensor)

    return input_tensor


# ─────────────────────────────────────────────────────────────────────────────
# HUMIDITY  (HSV fallback)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_flir(image):

    image_bgr = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR,
    )

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    return image_bgr, hsv