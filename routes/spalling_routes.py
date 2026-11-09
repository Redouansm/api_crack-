from fastapi import APIRouter, Request

from services.spalling_detection_service import process_spalling_detection
from utils.multipart_response import process_images

router = APIRouter()


@router.post("/predict")
async def predict_spalling(request: Request):

    return await process_images(request, process_spalling_detection, "no_spalling", tag="spalling")
