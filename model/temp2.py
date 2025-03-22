from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import logging
from inference.OpenAI_inference import image_classification_llm
from inference.MobileNet_inference import image_classification_mobilenet
import asyncio
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Classifier Model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENT_CLASSES = {
    "alb_id": "ID Card of Albania",
    "aze_passport": "Passport of Azerbaijan",
    "esp_id": "ID Card of Spain",
    "est_id": "ID Card of Estonia",
    "fin_id": "ID Card of Finland",
    "grc_passport": "Passport of Greece",
    "lva_passport": "Passport of Latvia",
    "rus_internalpassport": "Internal passport of Russia",
    "srb_passport": "Passport of Serbia",
    "svk_id": "ID Card of Slovakia"
}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        logger.info(f"Received combined prediction request for file: {image.filename}")
        
        # Read image content
        image_content = await image.read()
        
        # Run both models in parallel - pass the binary content directly
        llm_task = image_classification_llm(image_content, list(DOCUMENT_CLASSES.values()))
        mobilenet_task = image_classification_mobilenet(image_content, DOCUMENT_CLASSES)
        
        results = await asyncio.gather(llm_task, mobilenet_task)
        
        llm_result, mobilenet_result = results
        
        logger.info(f"Combined prediction: LLM: {llm_result}, MobileNet: {mobilenet_result}")
        
        return {
            "llm": {
                "class_name": llm_result,
                "confidence": 1.0
            },
            "mobilenet": {
                "class_name": mobilenet_result,
                "confidence": 1.0
            }
        }
    
    except Exception as e:
        logger.exception("Error during combined prediction")
        raise

if __name__ == "__main__":
    import uvicorn
    
    # Simple mock class for testing
    class MockFile:
        async def read(self): 
            return Path("data/processed/val/alb_id/38.jpg").read_bytes()
    
    # Quick test before starting server
    async def test():
        mock = MockFile()
        llm = await image_classification_llm(mock, list(DOCUMENT_CLASSES.values()))
        mob = await image_classification_mobilenet(await mock.read(), DOCUMENT_CLASSES)
        print(f"LLM: {llm}, MobileNet: {mob}")
    
    # Run test and start server
    asyncio.run(test())
    uvicorn.run(app, host="0.0.0.0", port=8000)

