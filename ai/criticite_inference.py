import torch
import numpy as np
from torchvision import transforms
from ai.criticite_model_loader import load_model

# =====================================
# CLASSES
# =====================================

CLASS_NAMES = {
    0: "Haute",
    1: "Modérée",
    2: "Très Haute"
}

# =====================================
# TTA TRANSFORM
# =====================================

TTA_TRANSFORM = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05
    ),

    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 0.5)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

VAL_TRANSFORM = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# INFERENCE
# =====================================

def predict_criticite(
    image,
    n_aug=10
):

    loaded = load_model("mobilenetv2_best")


    backend = loaded["backend"]

    model   = loaded["model"]

    probs_list = []

    # =====================================
    # TTA PASSES
    # =====================================

    for _ in range(n_aug):

        tensor = TTA_TRANSFORM(image).unsqueeze(0)

        if backend == "cuda":

            tensor = tensor.half().to("cuda")

            with torch.inference_mode():

                out = model(tensor)

                prob = torch.softmax(
                    out, dim=1
                ).squeeze().float().cpu().numpy()

        elif backend == "openvino":

            x = tensor.numpy().astype(np.float32)

            result = model([x])

            logits = list(result.values())[0].squeeze()

            prob = (
                np.exp(logits) /
                np.exp(logits).sum()
            )

        else:

            with torch.inference_mode():

                out = model(tensor)

                prob = torch.softmax(
                    out, dim=1
                ).squeeze().cpu().numpy()

        probs_list.append(prob)

    # =====================================
    # BASE PASS (sans augmentation)
    # =====================================

    tensor_base = VAL_TRANSFORM(image).unsqueeze(0)

    if backend == "cuda":

        tensor_base = tensor_base.half().to("cuda")

        with torch.inference_mode():

            out = model(tensor_base)

            prob = torch.softmax(
                out, dim=1
            ).squeeze().float().cpu().numpy()

    elif backend == "openvino":

        x = tensor_base.numpy().astype(np.float32)

        result = model([x])

        logits = list(result.values())[0].squeeze()

        prob = (
            np.exp(logits) /
            np.exp(logits).sum()
        )

    else:

        with torch.inference_mode():

            out = model(tensor_base)

            prob = torch.softmax(
                out, dim=1
            ).squeeze().cpu().numpy()

    probs_list.append(prob)

    # =====================================
    # MOYENNE TTA
    # =====================================

    final_probs = np.mean(probs_list, axis=0)

    pred_idx    = int(final_probs.argmax())

    pred_class  = CLASS_NAMES[pred_idx]

    confidence  = round(float(final_probs[pred_idx]), 3)

    return {
        "Criticité":     pred_class,
        "confidence": confidence,
    #   "probabilites": {
    #        CLASS_NAMES[i]: round(float(p), 3)
    #        for i, p in enumerate(final_probs)
    #    }
    }

