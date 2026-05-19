from torchvision import transforms
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess_for_criticite(
    image,
    size=224
):

    transform = transforms.Compose([

        transforms.Resize((size, size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.numpy()

    input_tensor = input_tensor.astype(
        np.float32
    )

    input_tensor = np.ascontiguousarray(
        input_tensor
    )

    return input_tensor