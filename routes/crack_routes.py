from fastapi import APIRouter, Request

from services.crack_detection_service import process_crack_detection
from utils.multipart_response import process_images

router = APIRouter()


@router.post("/predict")
async def predict_crack(request: Request):

    return await process_images(request, process_crack_detection, "no_crack", tag="crack")
