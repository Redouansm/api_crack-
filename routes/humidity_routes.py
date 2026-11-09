from fastapi import APIRouter, Request

from services.humidity_detection_service import process_humidity_detection
from utils.multipart_response import process_images, to_float

router = APIRouter()


@router.post("/predict")
async def predict_humidity(request: Request):

    return await process_images(
        request,
        process_humidity_detection,
        "no_humidity",
        extra_fields={"temp_air": to_float, "humidite_relative": to_float},
        tag="humidity",
    )
