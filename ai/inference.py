import cv2
import numpy as np
import torch

from ai.model_loader import load_model
def predict(
    input_tensor,
    model_name
):

    loaded = load_model(model_name)

    backend = loaded["backend"]

    model = loaded["model"]

    # =====================================
    # CUDA
    # =====================================

    if backend == "cuda":

        input_tensor = torch.tensor(
            input_tensor
        ).cuda()

        with torch.no_grad():

            output = model(input_tensor)

        return output.cpu().numpy()

    # =====================================
    # OPENVINO
    # =====================================

    elif backend == "openvino":

        try:

            infer_request = (
                model.create_infer_request()
            )

            results = infer_request.infer({
                0: input_tensor
            })

            output = results[
                model.output(0)
            ]

            return output

        except Exception as e:

            print(
                "OpenVINO failed:",
                e
            )

            print(
                "Fallback to CPU"
            )

            pytorch_model = loaded[
                "pytorch_model"
            ]

            input_tensor = torch.tensor(
                input_tensor
            )

            with torch.no_grad():

                output = pytorch_model(
                    input_tensor
                )

            return output.numpy()

    # =====================================
    # CPU
    # =====================================

    else:

        input_tensor = torch.tensor(
            input_tensor
        )

        with torch.no_grad():

            output = model(input_tensor)

        return output.numpy()
    



#------------------------------------------------------------------------------#
#                                   HUMIDITY                                   #          
#------------------------------------------------------------------------------#

# ─────────────────────────────────────────────────────────────────────────────
# HUMIDITY — détection depuis image thermique FLIR (palette Iron)
# ─────────────────────────────────────────────────────────────────────────────
#
# Palette Iron FLIR C5 :
#   Chaud  → jaune/orange/rouge  (H: 0-30,  V: élevé)
#   Froid  → violet/bleu/noir    (H: 100-165, V: faible)
#
# Les zones humides absorbent l'humidité et sont plus froides que
# les surfaces sèches environnantes → elles apparaissent en violet/bleu.
#
# Deux masques :
#   M1 : violet/bleu froid  H:100-165, S:>60, V:<160
#   M2 : très froid/noir    H:0-30,    S:>150, V:<100  (extrême froid Iron)
# ─────────────────────────────────────────────────────────────────────────────
 
def predict_humidity_hsv(
    hsv,
    palette="iron",
    sensitivity="medium",
    min_area=None,
):
 
    img_h, img_w = hsv.shape[:2]
    img_area     = img_h * img_w
 
    # ── Masques couleur froid (palette Iron) ─────────────────────────────────
    # M1 : violet → bleu (zones humides typiques)
    m1 = cv2.inRange(hsv, np.array([100, 60, 10]), np.array([165, 255, 160]))
    # M2 : rouge très sombre → noir (zones très froides)
    m2 = cv2.inRange(hsv, np.array([0, 150, 10]), np.array([30, 255, 100]))
 
    mask = cv2.bitwise_or(m1, m2)
 
    # ── Exclure les éléments UI FLIR ─────────────────────────────────────────
    # Haut 12%  : température + logo FLIR
    # Gauche 15%: barre de palette + température min/max
    # Droite 10%: artefacts bord
    # Bas 8%    : artefacts bord bas
    mask[:int(img_h * 0.12), :]    = 0
    mask[int(img_h * 0.92):, :]    = 0
    mask[:, :int(img_w * 0.15)]    = 0
    mask[:, int(img_w * 0.90):]    = 0
 
    # ── Morphologie : fermer les trous, supprimer le bruit ───────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
 
    # ── Contours ─────────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
 
    # Taille minimum : 1% de l'image (robuste aux différentes résolutions)
    area_min = max(
        min_area if min_area is not None else 0,
        int(img_area * 0.01),
    )
 
    zones = []
 
    for cnt in contours:
 
        area = cv2.contourArea(cnt)
 
        if area < area_min:
            continue
 
        x, y, w, h = cv2.boundingRect(cnt)
 
        roi_mask = mask[y:y+h, x:x+w]
        density  = np.sum(roi_mask > 0) / max(w * h, 1)
 
        zones.append({
            "bbox":       (x, y, w, h),
            "area":       int(area),
            "confidence": round(float(density), 3),
            "contour":    cnt,
        })
 
    zones.sort(key=lambda z: z["area"], reverse=True)
 
    # ── Filtrer les micro-zones vs la plus grande ─────────────────────────────
    # Toute zone < 25% de la plus grande est du bruit (ex: petites boxes en haut)
    if len(zones) > 1:
        max_area = zones[0]["area"]
        zones    = [z for z in zones if z["area"] >= max_area * 0.25]
 
    return zones, mask