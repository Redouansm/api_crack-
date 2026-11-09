import os

import numpy as np

from ai.device import LazyModel, model_path, torch_device

WEIGHTS = os.getenv("CHUTE_WEIGHTS", model_path("chute", "best.pt"))
IMGSZ   = int(os.getenv("CHUTE_IMGSZ", "960"))          # taille d'entraînement
CONF    = float(os.getenv("CHUTE_CONF", "0.25"))
IOU     = float(os.getenv("CHUTE_IOU", "0.7"))          
MAX_DET = int(os.getenv("CHUTE_MAX_DET", "1000"))        
USE_TTA = os.getenv("CHUTE_TTA", "0") == "1"             


def _load():
    from ultralytics import YOLO

    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"Poids YOLOv12 introuvables : {WEIGHTS}")

    device = 0 if torch_device() == "cuda" else "cpu"
    model  = YOLO(WEIGHTS, task="detect")
    print(f"[chute] YOLOv12 ({WEIGHTS}) sur {'GPU' if device == 0 else 'CPU'}, "
          f"imgsz={IMGSZ}, conf={CONF}, classes={model.names}")
    return {"model": model, "device": device, "half": device == 0}


CHUTE_MODEL = LazyModel("chute", _load)


def predict_chutes(image_rgb: np.ndarray) -> list:
    m = CHUTE_MODEL.get()

    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])

    with CHUTE_MODEL.lock:
        result = m["model"].predict(
            image_bgr,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            augment=USE_TTA,
            device=m["device"],
            half=m["half"],
            verbose=False,
        )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss  = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names

    detections = [
        {
            "classe":     names.get(int(c), str(c)) if isinstance(names, dict) else str(c),
            "confidence": float(s),
            "bbox_xyxy":  tuple(int(round(v)) for v in b),
        }
        for b, s, c in zip(boxes, confs, clss)
    ]
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections
