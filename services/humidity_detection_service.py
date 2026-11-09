import cv2
import numpy as np
from PIL import Image

from ai.humidity_model import detecter_zones_froides, merge_overlapping_bboxes, predict_humidity_hsv
from services.humidity_criticality_service import (
    dessiner_zones,
    dessiner_zones_hsv,
    evaluer_zones,
    point_de_rosee,
    resume_criticite,
    resume_criticite_hsv,
)
from utils.date_utils import resolve_date_taken
from utils.flir_utils import (
    aligner_photo_visible,
    conditions_ambiantes,
    extract_embedded_visible,
    extract_metadata,
    extract_raw_thermal,
    temperature_celsius,
)
from utils.image_utils import rgb_to_jpeg_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal : FLIR radiométrique -> FastSAM -> criticité
# ─────────────────────────────────────────────────────────────────────────────

def analyser_flir(image_bytes: bytes, temp_air=None, humidite_relative=None):
    """
    Retourne (image_annotée_rgb, résultat) ou None si aucune humidité,
    ou False si l'image n'est pas un JPEG FLIR radiométrique.
    """

    meta   = extract_metadata(image_bytes)
    raw    = extract_raw_thermal(image_bytes)
    temp_c = temperature_celsius(raw, meta) if raw is not None else None

    if temp_c is None:
        return False

    # 1. Détection des zones froides (FastSAM ∩ bleu) sur la grille agrandie
    detection = detecter_zones_froides(temp_c)
    grid_h, grid_w = detection["temp_big"].shape

    # 2. Criticité (ΔT local + point de rosée + surface)
    t_air, hr, source = conditions_ambiantes(meta, temp_air, humidite_relative)
    t_rosee = point_de_rosee(t_air, hr)
    zones   = evaluer_zones(detection, t_rosee=t_rosee, source_rosee=source)

    if not zones:
        return None

    # 3. Fond d'image : photo visible ALIGNÉE à la même taille que la grille
    visible = extract_embedded_visible(image_bytes)
    if visible is not None:
        fond, aligne = aligner_photo_visible(visible, meta, (grid_w, grid_h))
        type_fond = "photo_visible_alignee" if aligne else "photo_visible_non_alignee"
    else:
        fond      = cv2.cvtColor(detection["colored_bgr"], cv2.COLOR_BGR2RGB)
        type_fond = "carte_thermique"

    annotated = dessiner_zones(fond, zones)

    result = resume_criticite(zones, t_air, hr, t_rosee, source)
    result["detection"] = {
        "sam_utilise": detection["sam_used"],
        "nb_segments": detection["nb_segments"],
        "fond_image":  type_fond,
        "taille":      [grid_w, grid_h],
    }

    return annotated, result


# ─────────────────────────────────────────────────────────────────────────────
# Repli : image non radiométrique -> couleurs de la palette Iron (HSV)
# ─────────────────────────────────────────────────────────────────────────────

def analyser_hsv(image: Image.Image):
    zones, image_bgr = predict_humidity_hsv(np.array(image))
    zones            = merge_overlapping_bboxes(zones, margin=40)

    if not zones:
        return None

    annotated = cv2.cvtColor(dessiner_zones_hsv(image_bgr, zones), cv2.COLOR_BGR2RGB)
    result    = resume_criticite_hsv(zones)
    result["detection"] = {"methode": "hsv_sans_temperature"}

    return annotated, result


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée appelé par la route
# ─────────────────────────────────────────────────────────────────────────────

def process_humidity_detection(
    image: Image.Image,
    image_bytes: bytes = None,
    filename: str = "",
    date_client: str = "",
    temp_air: float = None,
    humidite_relative: float = None,
):

    date_taken = resolve_date_taken(date_client, filename, image_bytes)

    sortie = analyser_flir(image_bytes, temp_air, humidite_relative) if image_bytes else False

    if sortie is False:                       # pas de températures -> repli HSV
        sortie = analyser_hsv(image)

    if sortie is None:
        return None

    annotated, result = sortie
    return rgb_to_jpeg_buffer(annotated), {"date_taken": date_taken, **result}
