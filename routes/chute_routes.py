from fastapi import APIRouter, Request

from services.chute_detection_service import process_chute_detection
from utils.multipart_response import process_images

router = APIRouter()


@router.post("/predict")
async def predict_chute(request: Request):

    return await process_images(request, process_chute_detection, "no_chute", tag="chute")
