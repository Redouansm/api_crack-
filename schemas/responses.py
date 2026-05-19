from pydantic import BaseModel
from typing import Optional, Dict

class CriticiteResult(BaseModel):

    classe: str

    confidence: float

   # probabilites: Dict[str, float]


class DetectionFileResponse(BaseModel):

    image_id: str

    filename: str

    content_type: str = "image/jpeg"

    criticite: Optional[CriticiteResult] = None