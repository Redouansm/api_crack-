from PIL import Image
import io

from ai.criticite_inference import predict_criticite

# =====================================
# PROCESS CRITICITE
# =====================================

def process_criticite(
    image_rgb
):

    # =====================================
    # CONVERT TO PIL
    # =====================================

    if not isinstance(image_rgb, Image.Image):

        image_pil = Image.fromarray(image_rgb)

    else:

        image_pil = image_rgb

    # =====================================
    # INFERENCE
    # =====================================

    result = predict_criticite(
        image=image_pil,
        n_aug=10
    )

    # =====================================
    # RETURN
    # =====================================

    return {
        "classe":     result["classe"],
        "confidence": result["confidence"]
    }