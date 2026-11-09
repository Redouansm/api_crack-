# AI Inspection — API d'inférence

API FastAPI qui détecte 4 types de désordres sur des photos de bâtiment :

| Route | Modèle | Sortie |
|---|---|---|
| `POST /crack-detection/predict` | SegFormer mit-b3 (smp) + MobileNetV2 criticité | masque rouge + criticité |
| `POST /humidity-detection/predict` | FLIR radiométrique + FastSAM | zones + criticité (°C) |
| `POST /chute-detection/predict` | YOLOv12-l (ultralytics) | boîtes, détection seule |
| `POST /spalling-detection/predict` | SegFormer mit-b2 (transformers) | masque magenta, détection seule |

`GET /health` indique le matériel utilisé et l'état de chaque modèle.

# Structure

```txt
ai-inspection-inference/
├── main.py                              # routes + préchargement des modèles
├── Dockerfile
├── docker-compose.yml                   # GPU directement
├── requirements.txt
├── convert_model.py                     # export OpenVINO (fissures)
│
├── routes/                              # une route = 10 lignes, logique commune dans utils/multipart_response.py
│   ├── crack_routes.py
│   ├── humidity_routes.py
│   ├── chute_routes.py
│   └── spalling_routes.py
│
├── services/                            # logique métier : détection -> résultat JSON + image annotée
│   ├── crack_detection_service.py
│   ├── crack_criticality_service.py
│   ├── humidity_detection_service.py
│   ├── humidity_criticality_service.py
│   ├── chute_detection_service.py       # détection seule
│   └── spalling_detection_service.py    # détection seule
│
├── ai/                                  # chargement des modèles + inférence
│   ├── device.py                        # CUDA / OpenVINO / CPU, chargement paresseux, verrous
│   ├── crack_model.py
│   ├── crack_criticality_model.py
│   ├── humidity_model.py
│   ├── chute_model.py                   # YOLOv12
│   └── spalling_model.py                # SegFormer mit-b2
│
├── utils/
│   ├── date_utils.py                    # date : client > EXIF > nom de fichier > maintenant
│   ├── image_utils.py
│   ├── multipart_response.py            # lecture du formulaire + réponse multipart
│   └── flir_utils.py
│
└── models/                              # poids (non versionnés)
    ├── crack/        best_model_b3.pth, mobilenetv2_best.pth (+ .xml/.bin OpenVINO)
    ├── humidity/     FastSAM-s.pt (téléchargé au build)
    ├── chute/        best.pt
    └── spalling/     best_segformer_b2_v2.pth
```

# Poids des modèles

| Fichier | Provenance |
|---|---|
| `models/crack/best_model_b3.pth` | entraînement fissures |
| `models/crack/mobilenetv2_best.pth` | entraînement criticité |
| `models/chute/best.pt` | notebook `yolo12_det_colab` → `yolo12_chute/yolo_runs/yolo12_det_chute/weights/best.pt` |
| `models/spalling/best_segformer_b2_v2.pth` | notebook `seg2_v2` → `spalling_models/best_segformer_b2_v2.pth` |

Si un fichier manque, l'API démarre quand même : seule la route concernée échoue
(voir `/health`). Chemins modifiables par `CHUTE_WEIGHTS`, `SPALLING_WEIGHTS`, `MODELS_DIR`.

Le SegFormer spalling est reconstruit à partir des formes du fichier `.pth` :
aucun accès à huggingface.co n'est nécessaire.

# Lancer

## Docker (GPU NVIDIA)

Prérequis : driver NVIDIA + NVIDIA Container Toolkit.

```bash
docker compose up -d --build
```

API sur `http://localhost:8888`, Swagger sur `http://localhost:8888/docs`.

## Local

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Sans GPU NVIDIA (OpenVINO, fissures uniquement)

```bash
python convert_model.py
```

Chutes et spalling tournent alors en PyTorch CPU (plus lent, surtout YOLOv12-l à 960 px).

# Contrat commun des 4 routes

## Requête — `multipart/form-data`

| Champ | Obligatoire | Description |
|---|---|---|
| `<image_id>` | oui | fichier image (un ou plusieurs) |
| `<nom_fichier>__date__` | non | date de prise de vue |
| `temp_air`, `humidite_relative` | non | humidité uniquement (voir plus bas) |

## Réponse — `multipart/form-data; boundary=...`

Défaut détecté :

```txt
<image_id>             image/jpeg         <nom>_segmented.jpg
<image_id>_criticite   application/json   résultat
```

Rien détecté :

```txt
<image_id>             text/plain         no_crack | no_humidity | no_chute | no_spalling
```

Une image illisible ou en erreur est ignorée (journalisée côté serveur).

Les champs `date_taken`, `classe`, `criticite`, `confidence` sont présents pour les 4 routes.
Pour **chute** et **spalling** (détection seule, pas encore de modèle de criticité),
`criticite` vaut `null`.

# Chute Detection — YOLOv12

```txt
POST /chute-detection/predict
```

Carreaux de zellige manquants (classe `missing_tile`). Mêmes réglages que le notebook :
`imgsz=960`, `conf=0.25`, `max_det=1000`. Validation : mAP50 0,83 · mAP50-95 0,53 · P 0,92 · R 0,71.

```json
{
  "date_taken": "2026:11:09 10:12:00",
  "classe": "chute",
  "criticite": null,
  "confidence": 0.912,
  "nb_elements": 3,
  "surface_pct": 4.81,
  "zones": [
    {"zone": 1, "classe": "missing_tile", "confidence": 0.912, "bbox": [412, 220, 64, 61], "surface_pct": 1.9}
  ],
  "detection": {"modele": "yolov12", "imgsz": 960, "conf_min": 0.25}
}
```

`bbox` = `[x, y, largeur, hauteur]` en pixels de l'image envoyée (comme l'humidité).
`surface_pct` global = union des boîtes.

Variables : `CHUTE_CONF`, `CHUTE_IOU`, `CHUTE_IMGSZ`, `CHUTE_MAX_DET`, `CHUTE_TTA` (1 = +1-2 pts de rappel, ~3× plus lent).

# Spalling Detection — SegFormer mit-b2

```txt
POST /spalling-detection/predict
```

Prétraitement identique au `val_tf` du notebook (resize 512×512, normalisation ImageNet),
TTA flip horizontal, masque remis à la taille d'origine. Les zones de moins de 0,05 % de l'image sont ignorées.

```json
{
  "date_taken": "2026:11:09 10:12:00",
  "classe": "spalling",
  "criticite": null,
  "confidence": 0.874,
  "nb_zones": 2,
  "surface_pct": 6.3,
  "zones": [
    {"zone": 1, "confidence": 0.881, "bbox": [120, 340, 410, 220], "surface_pct": 5.1}
  ],
  "detection": {"modele": "segformer_mit_b2", "seuil": 0.5}
}
```

`confidence` = probabilité moyenne du modèle sur les pixels détectés.

Variables : `SPALLING_THRESHOLD`, `SPALLING_TTA`, `SPALLING_MIN_ZONE_PCT`, `SPALLING_MORPHO_CLOSE`.

# Crack Detection

```txt
POST /crack-detection/predict
```

GPU : sliding-window HD + TTA. CPU / OpenVINO : image réduite à 448 px.

```json
{"date_taken": "2026:04:30 09:44:00", "classe": "crack", "criticite": "MOYEN", "confidence": 0.83}
```

Variables : `CRACK_GPU_TTA`, `CRACK_GPU_BATCH`, `CRACK_GPU_OVERLAP`, `CRACK_GPU_THRESHOLD`,
`CRACK_GPU_MIN_AREA`, `CRACK_MIN_PIXELS`, `CRACK_CRITICALITY_TTA`.

# Variables générales

| Variable | Défaut | Rôle |
|---|---|---|
| `AI_DEVICE` | auto | forcer `cuda`, `openvino_gpu`, `openvino_cpu` ou `cpu` |
| `PRELOAD_MODELS` | `1` | charger tous les modèles au démarrage |
| `MODELS_DIR` | `./models` | dossier des poids |

---

# Humidity Detection — FastSAM + criticité

## Route

```txt
POST /humidity-detection/predict
```

## Champs du formulaire (multipart/form-data)

| Champ | Obligatoire | Description |
|---|---|---|
| `<image_id>` | oui | JPEG FLIR radiométrique (un ou plusieurs) |
| `<nom_fichier>__date__` | non | date de prise de vue (comme crack) |
| `temp_air` | non | température de l'air mesurée sur site (°C) |
| `humidite_relative` | non | humidité relative mesurée sur site (%) |

`temp_air` + `humidite_relative` rendent le calcul du risque de condensation fiable.
Sans eux, on utilise les valeurs des métadonnées FLIR (souvent 20 °C / 50 % usine)
et ce critère compte moitié moins.

## Pipeline

1. Températures en °C depuis le RawThermalImage (Planck, correction automatique de l'ordre des octets).
2. Carte agrandie x4 (160x120 -> 640x480), FastSAM propose des segments (GPU si disponible).
3. Zones = segments SAM ∩ moitié froide de l'image (repli : moitié froide seule).
4. Photo visible embarquée alignée sur la thermique (Real2IR, OffsetX/Y) à la même taille 640x480.
5. Criticité par zone, résultat dessiné sur la photo visible.

## Criticité (score 0-100 par zone)

| Critère | Points | Calcul |
|---|---|---|
| ΔT local | 0-60 | T mur sain autour − T zone ; 3 °C ou plus = 60 pts. Zone ignorée si ΔT < 0,5 °C |
| Condensation | 0-20 | marge = T la plus froide de la zone − point de rosée ; ≤ 0 °C = 20 pts, ≥ 3 °C = 0 |
| Étendue | 0-20 | % de l'image ; 25 % ou plus = 20 pts |

Score ≥ 60 -> **CRITIQUE**, ≥ 35 -> **MOYEN**, sinon **FAIBLE**. La criticité de l'image est celle de la pire zone.
Si aucune zone n'a un ΔT suffisant, l'API répond `no_humidity`.

Seuils réglables par variables d'environnement : `HUMIDITY_DELTA_MIN`, `HUMIDITY_DELTA_MAX`,
`HUMIDITY_SURFACE_MAX`, `HUMIDITY_SEUIL_CRITIQUE`, `HUMIDITY_SEUIL_MOYEN`, `HUMIDITY_UPSCALE`,
`HUMIDITY_SAM_CONF`, `HUMIDITY_SAM_WEIGHTS`.

## Exemple de JSON `<image_id>_criticite`

```json
{
  "date_taken": "2026:05:12 10:31:04",
  "classe": "humidite",
  "criticite": "MOYEN",
  "confidence": 0.86,
  "score": 41,
  "conditions": {"temp_air": 22.0, "humidite_relative": 70.0, "point_rosee": 16.3, "source": "client"},
  "zones": [
    {"zone": 1, "criticite": "MOYEN", "score": 41, "confidence": 0.86, "delta_t": 1.9,
     "temp_zone": 23.0, "temp_min_zone": 22.4, "temp_reference": 24.9,
     "marge_rosee": 6.1, "surface_pct": 4.2, "bbox": [165, 236, 71, 93]}
  ],
  "detection": {"sam_utilise": true, "nb_segments": 24, "fond_image": "photo_visible_alignee", "taille": [640, 480]}
}
```

Les champs `date_taken`, `classe`, `criticite`, `confidence` sont les mêmes qu'avant ;
`score`, `conditions`, `zones` et `detection` sont nouveaux.
