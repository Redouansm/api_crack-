import subprocess
import tempfile
import json
import io
import os
import re

import numpy as np

from PIL import Image
from datetime import datetime
from scipy import ndimage
from scipy.ndimage import uniform_filter


# ─────────────────────────────────────────────────────────────────────────────
# EXIFTOOL
# ─────────────────────────────────────────────────────────────────────────────

EXIFTOOL_CANDIDATES = [
    "exiftool",
    r"C:\Users\smahri\Downloads\exiftool-13.58_64\exiftool-13.58_64\exiftool.exe",
]


def find_exiftool():

    for candidate in EXIFTOOL_CANDIDATES:

        try:

            r = subprocess.run(
                [candidate, "-ver"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if r.returncode == 0:
                return candidate

        except (FileNotFoundError, OSError):
            continue

    return None


EXIFTOOL = find_exiftool()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def parse_float(value, default=0.0):

    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    match = re.search(r"-?[\d.]+", str(value))

    return float(match.group()) if match else default


# ─────────────────────────────────────────────────────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata(image_bytes: bytes) -> dict:

    if EXIFTOOL is None:
        return {}

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:

        r = subprocess.run(
            [EXIFTOOL, "-j", "-a", "-u", "-G", tmp_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )

        if not r.stdout.strip():
            return {}

        raw = json.loads(r.stdout)[0]

        meta = {}

        for key, val in raw.items():
            clean_key = key.split(":")[-1] if ":" in key else key
            meta[clean_key] = val

        return meta

    except Exception:
        return {}

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# RAW THERMAL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_raw_thermal(image_bytes: bytes) -> np.ndarray | None:

    if EXIFTOOL is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:

        r = subprocess.run(
            [EXIFTOOL, "-b", "-RawThermalImage", tmp_path],
            capture_output=True,
            timeout=15,
        )

        if not r.stdout or len(r.stdout) < 100:
            return None

        raw = np.array(
            Image.open(io.BytesIO(r.stdout)),
            dtype=np.float64,
        )

        return raw

    except Exception:
        return None

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED RGB EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_embedded_rgb(image_bytes: bytes, shape_thermique: tuple) -> np.ndarray | None:

    if EXIFTOOL is None:
        return None

    h, w = shape_thermique

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:

        for tag in ["-EmbeddedImage", "-PreviewImage", "-ThumbnailImage"]:

            r = subprocess.run(
                [EXIFTOOL, "-b", tag, tmp_path],
                capture_output=True,
                timeout=15,
            )

            if r.stdout and len(r.stdout) > 500:

                try:

                    img = Image.open(io.BytesIO(r.stdout)).convert("RGB")
                    img_resized = img.resize((w, h), Image.LANCZOS)

                    return np.array(img_resized)

                except Exception:
                    continue

        return None

    except Exception:
        return None

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# PLANCK CONVERSION  raw → °C
# ─────────────────────────────────────────────────────────────────────────────

def raw_to_celsius(raw: np.ndarray, meta: dict) -> np.ndarray | None:

    R1 = parse_float(meta.get("PlanckR1"), 0)
    R2 = parse_float(meta.get("PlanckR2"), 0)
    B  = parse_float(meta.get("PlanckB"),  0)
    F  = parse_float(meta.get("PlanckF"),  1)
    O  = parse_float(meta.get("PlanckO"),  0)

    if R1 == 0 or R2 == 0 or B == 0:
        return None

    denom    = R2 * (raw + O)
    safe_d   = np.where(denom > 0, denom, np.nan)
    log_arg  = R1 / safe_d + F
    safe_log = np.where(log_arg > 1, log_arg, np.nan)
    temp_c   = B / np.log(safe_log) - 273.15

    nb_nan = int(np.isnan(temp_c).sum())

    if nb_nan > 0:

        from scipy.ndimage import generic_filter

        def med_nan(v):
            v = v[~np.isnan(v)]
            return float(np.median(v)) if len(v) > 0 else np.nan

        temp_c = np.where(
            np.isnan(temp_c),
            generic_filter(temp_c, med_nan, size=3),
            temp_c,
        )

    return temp_c


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT PARAMS  (calcul thermique réel depuis pixels)
# ─────────────────────────────────────────────────────────────────────────────

def extract_params(meta: dict) -> dict:
    """
    Extrait les paramètres physiques depuis les métadonnées FLIR.

    t_atmo : depuis MakerNotes:ImageTemperatureMin (Kelvin) si disponible,
             sinon 22°C par défaut.
    hr_pct : 50% (FLIR C5 écrit toujours 50% usine).

    Point de rosée — formule Magnus :
        γ = ln(HR/100) + (a·T) / (b+T)
        Td = b·γ / (a−γ)   avec a=17.625, b=243.04
    """

    import math

    emissivite  = parse_float(meta.get("Emissivity"),                   0.95)
    temp_reflet = parse_float(meta.get("ReflectedApparentTemperature"), 20.0)
    distance    = parse_float(
        meta.get("SubjectDistance") or meta.get("ObjectDistance"),       1.0
    )

    hr_pct = 50.0

    t_min_raw = meta.get("ImageTemperatureMin")
    t_min_k   = parse_float(t_min_raw, None)
    if t_min_k is not None and t_min_k > 200:
        t_atmo = round(t_min_k - 273.15, 1)
    else:
        t_atmo = 22.0

    a       = 17.625
    b       = 243.04
    gamma   = math.log(max(hr_pct, 0.1) / 100.0) + (a * t_atmo) / (b + t_atmo)
    t_rosee = round((b * gamma) / (a - gamma), 1)

    return {
        "emissivite":  emissivite,
        "temp_reflet": temp_reflet,
        "distance":    distance,
        "humidite":    round(hr_pct / 100.0, 4),
        "temp_atmo":   t_atmo,
        "t_rosee":     t_rosee,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUSION DES BOUNDING BOXES PROCHES  (Union-Find)
# ─────────────────────────────────────────────────────────────────────────────

def fusionner_zones_proches(zones, temp_c, infos, params, distance_max=40):

    if len(zones) <= 1:
        return zones

    n      = len(zones)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    def bb_distance(a, b):
        dx = max(0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
        dy = max(0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))
        return max(dx, dy)

    def bb_iou(a, b):
        ix    = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        iy    = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        inter = ix * iy
        if inter == 0:
            return 0.0
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        return inter / (area_a + area_b - inter)

    for i in range(n):
        for j in range(i + 1, n):
            if bb_distance(zones[i], zones[j]) <= distance_max or bb_iou(zones[i], zones[j]) > 0.05:
                union(i, j)

    groupes = {}
    for i in range(n):
        groupes.setdefault(find(i), []).append(i)

    if len(groupes) == n:
        return zones

    tmoy        = infos["tmoy"]
    tstd        = infos["tstd"]
    mag_grad    = infos["mag_grad"]
    seuil_g     = infos["seuil_g"]
    delta_local = infos["delta_local"]
    idx_palette = infos["idx_palette"]

    nouvelles_zones = []

    for idx_groupe, (_, membres) in enumerate(groupes.items()):

        if len(membres) == 1:
            z = dict(zones[membres[0]])
            z["id"] = idx_groupe + 1
            nouvelles_zones.append(z)
            continue

        x1 = min(zones[m]["x1"] for m in membres)
        y1 = min(zones[m]["y1"] for m in membres)
        x2 = max(zones[m]["x2"] for m in membres)
        y2 = max(zones[m]["y2"] for m in membres)

        masque_union = np.zeros(temp_c.shape, dtype=bool)
        for m in membres:
            z     = zones[m]
            patch = infos["masque_final"][z["y1"]:z["y2"], z["x1"]:z["x2"]]
            masque_union[z["y1"]:z["y2"], z["x1"]:z["x2"]] |= patch

        tz      = temp_c[masque_union]
        nb_px   = int(masque_union.sum())
        t_min_z = float(np.nanmin(tz))  if nb_px > 0 else float(np.nanmin([zones[m]["temp_min"] for m in membres]))
        t_moy_z = float(np.nanmean(tz)) if nb_px > 0 else float(np.nanmean([zones[m]["temp_moy"] for m in membres]))
        delta   = tmoy - t_moy_z

        votes_zone        = {k: int(v[masque_union].sum()) for k, v in infos["masques"].items()}
        nb_masques_actifs = sum(1 for v in votes_zone.values() if v > nb_px * 0.3)

        _delta  = delta if (delta == delta) else 0.0
        _grad_m = float(np.nanmean(mag_grad[masque_union])) if nb_px > 0 else 0.0
        s_d   = min(40, int(_delta / max(tstd, 0.01) * 20))
        s_s   = min(30, int(np.log1p(nb_px) * 4))
        s_g   = min(30, int(_grad_m / max(seuil_g, 1e-9) * 15))
        score = s_d + s_s + s_g

        if   score >= 67: niveau, couleur = "CRITIQUE", (255, 34,  34)
        elif score >= 34: niveau, couleur = "MOYEN",    (255, 165, 0)
        else:             niveau, couleur = "FAIBLE",   (136, 255, 0)

        nouvelles_zones.append({
            "id":               idx_groupe + 1,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "pixels":           nb_px,
            "surface_pct":      nb_px / temp_c.size * 100,
            "temp_min":         t_min_z,
            "temp_moy":         t_moy_z,
            "delta":            delta,
            "delta_local_max":  float(delta_local[masque_union].max()) if nb_px > 0 else 0,
            "indice_froid_min": float(idx_palette[masque_union].min()) if nb_px > 0 else 0,
            "indice_froid_moy": float(idx_palette[masque_union].mean()) if nb_px > 0 else 0,
            "score":            score,
            "s_delta":          s_d,
            "s_surface":        s_s,
            "s_gradient":       s_g,
            "votes":            votes_zone,
            "nb_masques":       nb_masques_actifs,
            "niveau":           niveau,
            "couleur":          couleur,
            "confidence":       round(score / 100, 3),
            "fusionnee":        True,
            "nb_fusionnees":    len(membres),
        })

    nouvelles_zones.sort(key=lambda z: z["score"], reverse=True)

    return nouvelles_zones


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE MULTI-CRITÈRES  (5 masques + vote + fusion)
# ─────────────────────────────────────────────────────────────────────────────



def run_detection_pipeline(
    temp_c: np.ndarray,
    meta: dict,
    params: dict,
    sensibilite: float = 1.5,
    taille_min: int = 20,
    delta_local_min: float = 1.0,
    ratio_palette: float = 0.40,
    marge: int = 4,
):

    tmin = float(np.nanmin(temp_c))
    tmax = float(np.nanmax(temp_c))
    tmoy = float(np.nanmean(temp_c))
    tstd = float(np.nanstd(temp_c))

    masques = {}

    # ── M1 : Seuillage adaptatif ──────────────────────────────────────────────
    el_raw = meta.get("EstimatedLow")
    eh_raw = meta.get("EstimatedHigh")
    el = parse_float(el_raw, None) if el_raw is not None else None
    eh = parse_float(eh_raw, None) if eh_raw is not None else None

    if el is not None and el != 0:
        seuil = el + (tmoy - tmin) * 0.3
    else:
        seuil = tmoy - sensibilite * tstd

    if params["emissivite"] < 0.90:
        seuil += tstd * 0.2

    masques["seuil_adaptatif"] = (temp_c < seuil) & (~np.isnan(temp_c))

    # ── M2 : Delta local ──────────────────────────────────────────────────────
    temp_locale = uniform_filter(temp_c, size=30)
    delta_local = temp_locale - temp_c
    masques["delta_local"] = (delta_local > delta_local_min) & (~np.isnan(temp_c))

    # ── M3 : Ratio palette ────────────────────────────────────────────────────
    if el is not None and eh is not None and el != 0 and eh != 0:
        t_low, t_high = el, eh
    else:
        t_low, t_high = tmin, tmax

    idx_palette = np.clip(
        (temp_c - t_low) / max(t_high - t_low, 0.01),
        0, 1,
    )
    masques["palette_bleu"] = idx_palette < ratio_palette

    # ── M4 : Gradient thermique ───────────────────────────────────────────────
    gy, gx   = np.gradient(temp_c)
    mag_grad = np.sqrt(gx**2 + gy**2)
    seuil_g  = float(np.nanpercentile(mag_grad, 75))
    masques["gradient"] = (mag_grad > seuil_g) & (~np.isnan(temp_c))

    # ── M5 : Isotherme FLIR ───────────────────────────────────────────────────
    iso_en = meta.get("IsIsotherm1Enabled", "False")
    iso_t  = meta.get("IsothermTemp1", None)

    if str(iso_en).lower() in ("true", "1", "yes") and iso_t is not None:
        masques["isotherm_flir"] = (temp_c < parse_float(iso_t)) & (~np.isnan(temp_c))
    else:
        masques["isotherm_flir"] = np.zeros_like(temp_c, dtype=bool)

    # ── Vote majoritaire ──────────────────────────────────────────────────────
    vote          = sum(m.astype(int) for m in masques.values())
    masque_vote   = vote >= 3
    masque_propre = ndimage.binary_closing(masque_vote, iterations=2)
    masque_final  = ndimage.binary_opening(masque_propre, iterations=1)

    # ── Étiquetage ────────────────────────────────────────────────────────────
    distance      = max(params["distance"], 0.5)
    taille_min_ad = max(8, int(taille_min / distance))

    labels, nb_total = ndimage.label(masque_final)

    zones = []

    for i in range(1, nb_total + 1):

        region    = labels == i
        nb_pixels = int(region.sum())

        if nb_pixels < taille_min_ad:
            continue

        rows = np.where(region.any(axis=1))[0]
        cols = np.where(region.any(axis=0))[0]

        y1 = int(max(0,               rows.min() - marge))
        y2 = int(min(temp_c.shape[0], rows.max() + marge))
        x1 = int(max(0,               cols.min() - marge))
        x2 = int(min(temp_c.shape[1], cols.max() + marge))

        tz      = temp_c[region]
        t_moy_z = float(np.nanmean(tz))
        t_min_z = float(np.nanmin(tz))
        delta   = tmoy - t_moy_z

        votes_zone        = {k: int(v[region].sum()) for k, v in masques.items()}
        nb_masques_actifs = sum(1 for v in votes_zone.values() if v > nb_pixels * 0.3)

        _delta  = delta if (delta == delta) else 0.0  # NaN guard
        _grad_m = float(np.nanmean(mag_grad[region])) if region.any() else 0.0
        s_d   = min(40, int(_delta / max(tstd, 0.01) * 20))
        s_s   = min(30, int(np.log1p(nb_pixels) * 4))
        s_g   = min(30, int(_grad_m / max(seuil_g, 1e-9) * 15))
        score = s_d + s_s + s_g

        if   score >= 67: niveau, couleur = "CRITIQUE", (255, 34,  34)
        elif score >= 34: niveau, couleur = "MOYEN",    (255, 165, 0)
        else:             niveau, couleur = "FAIBLE",   (136, 255, 0)

        zones.append({
            "id":               len(zones) + 1,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "pixels":           nb_pixels,
            "surface_pct":      nb_pixels / temp_c.size * 100,
            "temp_min":         t_min_z,
            "temp_moy":         t_moy_z,
            "delta":            delta,
            "delta_local_max":  float(delta_local[region].max()),
            "indice_froid_min": float(idx_palette[region].min()),
            "indice_froid_moy": float(idx_palette[region].mean()),
            "score":            score,
            "s_delta":          s_d,
            "s_surface":        s_s,
            "s_gradient":       s_g,
            "votes":            votes_zone,
            "nb_masques":       nb_masques_actifs,
            "niveau":           niveau,
            "couleur":          couleur,
            "confidence":       round(score / 100, 3),
        })

    zones.sort(key=lambda z: z["score"], reverse=True)

    infos = {
        "seuil":        seuil,
        "seuil_g":      seuil_g,
        "mag_grad":     mag_grad,
        "delta_local":  delta_local,
        "idx_palette":  idx_palette,
        "masques":      masques,
        "vote":         vote,
        "masque_final": masque_final,
        "tmin": tmin, "tmax": tmax, "tmoy": tmoy, "tstd": tstd,
    }

    # ── Fusion des bounding boxes proches ─────────────────────────────────────
    zones = fusionner_zones_proches(zones, temp_c, infos, params, distance_max=15)

    return zones, infos