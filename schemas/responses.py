from pydantic import BaseModel


class DetectionFileResponse(BaseModel):

    image_id: str

    filename: str

    content_type: str = "image/jpeg"