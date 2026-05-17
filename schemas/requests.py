from pydantic import BaseModel


class CrackDetectionRequest(
    BaseModel
):

    threshold: float = 0.88

    model_name: str = "crack"