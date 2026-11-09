import math
import os

import cv2
import numpy as np

DELTA_MIN        = float(os.getenv("HUMIDITY_DELTA_MIN", "0.5"))    # °C
DELTA_MAX_PTS    = float(os.getenv("HUMIDITY_DELTA_MAX", "3.0"))    # °C -> 60 pts
SURFACE_MAX_PTS  = float(os.getenv("HUMIDITY_SURFACE_MAX", "25"))   # % -> 20 pts
MARGE_ROSEE_MAX  = 3.0                                              # °C -> 0 pt
SEUIL_CRITIQUE   = int(os.getenv("HUMIDITY_SEUIL_CRITIQUE", "60"))
SEUIL_MOYEN      = int(os.getenv("HUMIDITY_SEUIL_MOYEN", "35"))

POIDS_DELTA, POIDS_ROSEE, POIDS_SURFACE = 60, 20, 20

ORDRE   = {"CRITIQUE": 3, "MOYEN": 2, "FAIBLE": 1}
COULEUR = {                                   # RGB
    "CRITIQUE": (255, 34, 34),
    "MOYEN":    (255, 165, 0),
    "FAIBLE":   (136, 255, 0),
}


def point_de_rosee(t_air, hr_pct):
    """Formule de Magnus. Retourne None si les données manquent."""
    if t_air is None or hr_pct is None or hr_pct <= 0:
        return None
    a, b  = 17.625, 243.04
    gamma = math.log(min(hr_pct, 100.0) / 100.0) + (a * t_air) / (b + t_air)
    return (b * gamma) / (a - gamma)


def _temperature_reference(zone_mask, all_mask, temp, ring_px):

    def dilate(m, r):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(m.astype(np.uint8), k).astype(bool)

    zone_u8 = zone_mask.astype(np.uint8)
    ring    = dilate(zone_u8, 2 * ring_px) & ~dilate(zone_u8, ring_px // 2)
    ring   &= ~all_mask & np.isfinite(temp)

    if ring.sum() >= 50:
        return float(np.median(temp[ring])), "anneau"

    reste = ~all_mask & np.isfinite(temp)              # zone quasi plein cadre
    if reste.sum() >= 50:
        return float(np.median(temp[reste])), "reste_image"

    return None, "aucune"


def evaluer_zones(detection, t_rosee=None, source_rosee="absent", ring_px=None):

    temp     = detection["temp_big"]
    all_mask = detection["final_mask"]
    sam_used = detection["sam_used"]
    total_px = temp.size
    ring_px  = ring_px or max(8, int(0.06 * max(temp.shape)))      # ~40 px à 640

    zones = []

    for z in detection["zones"]:

        t_zone = temp[z["mask"] & np.isfinite(temp)]
        if t_zone.size == 0:
            continue

        t_ref, ref_type = _temperature_reference(z["mask"], all_mask, temp, ring_px)
        if t_ref is None:
            continue

        t_med   = float(np.median(t_zone))
        t_froid = float(np.percentile(t_zone, 5))
        delta   = t_ref - t_med

        if delta < DELTA_MIN:
            continue                                   # pas assez froide -> ignorée

        surface_pct = 100.0 * z["area"] / total_px

        s_delta   = POIDS_DELTA   * min(1.0, delta / DELTA_MAX_PTS)
        s_surface = POIDS_SURFACE * min(1.0, surface_pct / SURFACE_MAX_PTS)

        marge = None
        s_rosee = 0.0
        if t_rosee is not None:
            marge   = t_froid - t_rosee
            s_rosee = POIDS_ROSEE * float(np.clip((MARGE_ROSEE_MAX - marge) / MARGE_ROSEE_MAX, 0, 1))
            if source_rosee != "client":
                s_rosee *= 0.5                         # valeurs caméra peu fiables

        score = int(round(s_delta + s_rosee + s_surface))

        if   score >= SEUIL_CRITIQUE: niveau = "CRITIQUE"
        elif score >= SEUIL_MOYEN:    niveau = "MOYEN"
        else:                         niveau = "FAIBLE"

        confidence = 0.35 + 0.45 * min(1.0, delta / 2.5) + (0.2 if sam_used else 0.0)

        zones.append({
            **z,
            "niveau":          niveau,
            "score":           score,
            "confidence":      round(min(confidence, 0.99), 2),
            "delta_t":         round(delta, 2),
            "temp_zone":       round(t_med, 1),
            "temp_min_zone":   round(t_froid, 1),
            "temp_reference":  round(t_ref, 1),
            "reference":       ref_type,
            "surface_pct":     round(surface_pct, 1),
            "marge_rosee":     None if marge is None else round(marge, 1),
            "detail_score":    {
                "delta":   round(s_delta, 1),
                "rosee":   round(s_rosee, 1),
                "surface": round(s_surface, 1),
            },
        })

    zones.sort(key=lambda z: (ORDRE[z["niveau"]], z["score"]), reverse=True)
    for i, z in enumerate(zones, start=1):
        z["id"] = i

    return zones


def resume_criticite(zones, t_air=None, hr_pct=None, t_rosee=None, source_rosee="absent"):

    pire = zones[0]

    return {
        "classe":     "humidite",
        "criticite":  pire["niveau"],
        "confidence": pire["confidence"],
        "score":      pire["score"],
        "conditions": {
            "temp_air":          None if t_air  is None else round(t_air, 1),
            "humidite_relative": None if hr_pct is None else round(hr_pct, 1),
            "point_rosee":       None if t_rosee is None else round(t_rosee, 1),
            "source":            source_rosee,
        },
        "zones": [
            {
                "zone":           z["id"],
                "criticite":      z["niveau"],
                "score":          z["score"],
                "confidence":     z["confidence"],
                "delta_t":        z["delta_t"],
                "temp_zone":      z["temp_zone"],
                "temp_min_zone":  z["temp_min_zone"],
                "temp_reference": z["temp_reference"],
                "marge_rosee":    z["marge_rosee"],
                "surface_pct":    z["surface_pct"],
                "bbox":           list(z["bbox"]),
            }
            for z in zones
        ],
    }


def dessiner_zones(image_rgb, zones):

    out     = image_rgb.copy()
    overlay = image_rgb.copy()

    for z in zones:
        cv2.drawContours(overlay, [z["contour"]], -1, COULEUR[z["niveau"]], -1)
    out = cv2.addWeighted(out, 0.65, overlay, 0.35, 0)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, out.shape[1] / 1400)

    for z in zones:
        color      = COULEUR[z["niveau"]]
        x, y, w, h = z["bbox"]
        cv2.drawContours(out, [z["contour"]], -1, color, 2)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)

        label = f"{z['id']} {z['niveau']} -{z['delta_t']:.1f}C"
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        ty = y if y - th - 6 < 0 else y - th - 6
        cv2.rectangle(out, (x, ty), (x + tw + 6, ty + th + 6), color, -1)
        cv2.putText(out, label, (x + 3, ty + th + 2), font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# REPLI HSV (image sans températures) — criticité estimée sur la densité
# ─────────────────────────────────────────────────────────────────────────────

def _niveau_hsv(score):
    if score >= 67:
        return "CRITIQUE"
    if score >= 34:
        return "MOYEN"
    return "FAIBLE"


def resume_criticite_hsv(zones):

    zones_list = []
    for i, z in enumerate(zones, start=1):
        score = int(z["confidence"] * 100)
        zones_list.append({
            "zone":      i,
            "criticite": _niveau_hsv(score),
            "score":     score,
            "bbox":      [int(v) for v in z["bbox"]],
        })

    score_moyen = round(sum(z["score"] for z in zones_list) / len(zones_list)) if zones_list else 0
    pire        = max(zones_list, key=lambda z: ORDRE[z["criticite"]])["criticite"] if zones_list else "FAIBLE"

    return {
        "classe":     "humidite",
        "criticite":  pire,
        "confidence": round(score_moyen / 100.0, 2),
        "zones":      zones_list,
    }


def dessiner_zones_hsv(image_bgr, zones):

    out = image_bgr.copy()
    for i, z in enumerate(zones):
        x, y, w, h = z["bbox"]
        conf = z["confidence"]
        color = (0, 255, 100) if conf >= 0.70 else (0, 200, 255) if conf >= 0.45 else (80, 80, 255)

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)

        label = str(i + 1)
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.35
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.rectangle(out, (x, y), (x + tw + 4, y + th + 4), color, -1)
        cv2.putText(out, label, (x + 2, y + th + 2), font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    return out
