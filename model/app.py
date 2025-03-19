# model/app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import random
import uvicorn
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ID Document Classifier Model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# List of document classes
DOCUMENT_CLASSES = [
    "alb_id",             # ID Card of Albania
    "aze_passport",       # Passport of Azerbaijan
    "esp_id",             # ID Card of Spain
    "est_id",             # ID Card of Estonia
    "fin_id",             # ID Card of Finland
    "grc_passport",       # Passport of Greece
    "lva_passport",       # Passport of Latvia
    "rus_internalpassport", # Internal passport of Russia
    "srb_passport",       # Passport of Serbia
    "svk_id"              # ID Card of Slovakia
]

class Prediction(BaseModel):
    class_name: str
    confidence: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=Prediction)
async def predict(image: UploadFile = File(...)):
    try:
        # Log the request
        logger.info(f"Received prediction request for file: {image.filename}")
        
        # Read the image content - not used in this simplified version
        # content = await image.read()
        
        # In a real implementation, you would:
        # 1. Preprocess the image
        # 2. Run it through your ML model
        # 3. Return the prediction
        
        # For now, just return a random class
        random_class = random.choice(DOCUMENT_CLASSES)
        confidence = random.uniform(60.0, 99.9)  # Random confidence between 60% and 99.9%
        
        logger.info(f"Returning random prediction: {random_class} with confidence {confidence:.2f}%")
        
        return Prediction(
            class_name=random_class,
            confidence=confidence
        )
    
    except Exception as e:
        logger.exception("Error during prediction")
        raise

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
