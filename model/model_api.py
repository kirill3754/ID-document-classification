from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference.openai_inference import image_classification_llm
from inference.mobilenet_inference import image_classification_mobilenet
import asyncio
import config
import uvicorn
from loguru import logger


app = FastAPI(title="Classifier Model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        logger.info(f"Received: {image.filename}")
        img_bytes = await image.read()
        logger.info(f"Image read: {image.filename}")
        llm_task = asyncio.create_task(
            image_classification_llm(img_bytes, list(config.DOCUMENT_CLASSES.values()))
        )
        mobilenet_task = asyncio.create_task(
            image_classification_mobilenet(img_bytes, config.DOCUMENT_CLASSES)
        )
        (llm_class_name, llm_confidence), (
            mobilenet_class_name,
            mobilenet_confidence,
        ) = await asyncio.gather(llm_task, mobilenet_task)

        logger.info(
            f"Combined prediction results - LLM: {llm_class_name}, MobileNet: {mobilenet_class_name}"
        )
        return {
            "llm_prediction": {
                "class_name": llm_class_name,
                "confidence": llm_confidence,
            },
            "mobilenet_prediction": {
                "class_name": mobilenet_class_name,
                "confidence": mobilenet_confidence,
            },
        }
    except Exception as e:
        logger.exception("Error in predict")
        raise


if __name__ == "__main__":
    logger.info("Starting test server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
