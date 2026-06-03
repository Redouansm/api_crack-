from fastapi import FastAPI

from routes.crack_detection_routes import (
    router as crack_detection_router
)

from routes.humidity_detection_routes import (
    router as humidity_detection_router
)

app = FastAPI()

app.include_router(
    crack_detection_router,
    prefix="/crack-detection",
    tags=["Crack Detection"]
)

app.include_router(
    humidity_detection_router,
    prefix="/humidity-detection",
    tags=["Humidity Detection"]
)

@app.get("/")

def home():

    return {
        "message": "AI API running hhhhhhhhhhhhhhhhhhhhh"
            }