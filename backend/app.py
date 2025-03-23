# backend/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import Dict, Any
import os
from werkzeug.utils import secure_filename
from loguru import logger
import uvicorn
import shutil

app = FastAPI(title="Backend Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# CORS(app)

UPLOAD_FOLDER = "/tmp/uploads"
ALLOWED_EXTENSIONS = {"jpg"}
MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8000/predict")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/test")
async def health_check() -> Dict[str, str]:
    return {"status": "OK"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> Dict[str, Any]:
    if image is None:
        raise HTTPException(status_code=400, detail="No image provided")

    if image.filename == "":
        raise HTTPException(status_code=400, detail="No image selected")

    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    filepath = None
    try:
        filename = secure_filename(image.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        logger.info(f"Image saved to {filepath}")

        with open(filepath, "rb") as img_file:
            files = {"image": (filename, img_file, image.content_type)}
            response = requests.post(MODEL_SERVICE_URL, files=files)

        if response.status_code != 200:
            logger.error(f"Model service returned error: {response.text}")
            raise HTTPException(status_code=500, detail="Model service error")

        result = response.json()
        return result

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)
    logger.info("Starting test server on http://localhost:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
