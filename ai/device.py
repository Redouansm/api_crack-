import os
import threading
from functools import lru_cache

import torch

OPENVINO_AVAILABLE = False
try:
    from openvino import Core  
    OPENVINO_AVAILABLE = True
except Exception:
    try:
        from openvino.runtime import Core  # type: ignore
        OPENVINO_AVAILABLE = True
    except Exception:
        Core = None

APP_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(APP_DIR, "models"))

_VALID = {"cuda", "openvino_gpu", "openvino_cpu", "cpu"}


@lru_cache(maxsize=1)
def get_best_device() -> str:
    """Détecté une seule fois au démarrage, puis mis en cache."""

    forced = os.getenv("AI_DEVICE", "").strip().lower()
    if forced in _VALID:
        print(f"[device] forcé par AI_DEVICE : {forced}")
        return forced

    if torch.cuda.is_available():
        print(f"[device] NVIDIA GPU (CUDA) : {torch.cuda.get_device_name(0)}")
        return "cuda"

    if OPENVINO_AVAILABLE:
        try:
            devices = Core().available_devices
            print("[device] OpenVINO devices :", devices)
            return "openvino_gpu" if "GPU" in devices else "openvino_cpu"
        except Exception as e:
            print("[device] OpenVINO error :", e)

    print("[device] Torch CPU")
    return "cpu"


def torch_device() -> str:
    """Device PyTorch pour les modèles sans export OpenVINO (chute, spalling)."""
    return "cuda" if get_best_device() == "cuda" else "cpu"


def model_path(*parts: str) -> str:
    return os.path.join(MODELS_DIR, *parts)


def clean_state_dict(checkpoint) -> dict:

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in checkpoint.items()
    }


class LazyModel:

    def __init__(self, name: str, loader):
        self.name   = name
        self.loader = loader
        self.lock   = threading.Lock()       # inférence
        self._load  = threading.Lock()       # chargement
        self._value = None
        self._error = None

    def get(self):
        if self._value is not None:
            return self._value
        with self._load:
            if self._value is None:
                try:
                    self._value = self.loader()
                    self._error = None
                    print(f"[model] {self.name} chargé")
                except Exception as e:
                    self._error = f"{type(e).__name__}: {e}"
                    raise
        return self._value

    @property
    def status(self) -> str:
        if self._value is not None:
            return "loaded"
        return f"error: {self._error}" if self._error else "not_loaded"


def load_openvino(xml_path: str, device_type: str):

    if not OPENVINO_AVAILABLE:
        raise RuntimeError("OpenVINO non installé")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)

    core     = Core()
    ov_model = core.read_model(model=xml_path, weights=os.path.splitext(xml_path)[0] + ".bin")
    return core.compile_model(
        model=ov_model,
        device_name="GPU" if device_type == "openvino_gpu" else "CPU",
        config={"PERFORMANCE_HINT": "THROUGHPUT"},
    )


def load_torch_weights(model: torch.nn.Module, weights_path: str, device: str) -> torch.nn.Module:

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint))
    model.eval()
    if device == "cuda":
        model.half().to("cuda")
    return model


def load_torch_or_openvino(build_fn, weights_path: str, xml_path: str) -> dict:
    device_type = get_best_device()

    if device_type == "cuda":
        return {"backend": "cuda", "model": load_torch_weights(build_fn(), weights_path, "cuda")}

    if device_type in ("openvino_gpu", "openvino_cpu"):
        try:
            return {"backend": "openvino", "model": load_openvino(xml_path, device_type)}
        except Exception as e:
            print(f"[device] OpenVINO indisponible pour {os.path.basename(xml_path)} ({e}) -> PyTorch CPU")

    return {"backend": "cpu", "model": load_torch_weights(build_fn(), weights_path, "cpu")}
