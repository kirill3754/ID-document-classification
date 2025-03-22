from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from inference.OpenAI_inference import image_classification_llm
from inference.MobileNet_inference import image_classification_mobilenet
import asyncio
import os
from pathlib import Path
import uvicorn  # Add this import for the server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Classifier Model')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

DOCUMENT_CLASSES = {
    'alb_id': 'ID Card of Albania',
    'aze_passport': 'Passport of Azerbaijan',
    'esp_id': 'ID Card of Spain',
    'est_id': 'ID Card of Estonia',
    'fin_id': 'ID Card of Finland',
    'grc_passport': 'Passport of Greece',
    'lva_passport': 'Passport of Latvia',
    'rus_internalpassport': 'Internal passport of Russia',
    'srb_passport': 'Passport of Serbia',
    'svk_id': 'ID Card of Slovakia'
}

class Prediction(BaseModel):
    class_name: str
    confidence: float

class CombinedPrediction(BaseModel):
    llm: Prediction
    mobilenet: Prediction

@app.get('/health')
def health_check():
    return {'status': 'healthy'}

'''@app.post('/predict_llm', response_model=Prediction)
async def predict(image: UploadFile = File(...)):
    try:
        logger.info(f'Received LLM prediction request for file: {image.filename}')
        
        class_name = await image_classification_llm(image, list(DOCUMENT_CLASSES.values()))
        confidence = 1
        
        logger.info(f'LLM prediction: {class_name}')
        
        return Prediction(
            class_name=class_name,
            confidence=confidence
        )
    
    except Exception as e:
        logger.exception('Error during LLM prediction')
        raise

@app.post('/predict_mobilenet', response_model=Prediction)
async def predict_mobilenet(image: UploadFile = File(...)):
    try:
        logger.info(f'Received MobileNet prediction request for file: {image.filename}')
        
        class_name = await image_classification_mobilenet(image, DOCUMENT_CLASSES)
        confidence = 1
        
        logger.info(f'MobileNet returning prediction: {class_name} with confidence {confidence:.2f}%')
        
        return Prediction(
            class_name=class_name,
            confidence=confidence
        )
    
    except Exception as e:
        logger.exception('Error during MobileNet prediction')
        raise'''

@app.post('/predict', response_model=CombinedPrediction)
async def predict(image: UploadFile = File(...)):
    try:
        logger.info(f'Received: {image.filename}')
        content = await image.read()
        
        class FileWrapper:
            def __init__(self, content, filename):
                self.content = content
                self.filename = filename
            async def read(self):
                return self.content
        
        wrapper = FileWrapper(content, image.filename)
        llm_task = asyncio.create_task(image_classification_llm(wrapper, list(DOCUMENT_CLASSES.values())))
        mobilenet_task = asyncio.create_task(image_classification_mobilenet(wrapper, DOCUMENT_CLASSES))
        llm_class_name, mobilenet_class_name = await asyncio.gather(llm_task, mobilenet_task)
        
        logger.info(f'Combined prediction results - LLM: {llm_class_name}, MobileNet: {mobilenet_class_name}')
        return CombinedPrediction(
            llm=Prediction(class_name=llm_class_name, confidence=1),
            mobilenet=Prediction(class_name=mobilenet_class_name[0], confidence=mobilenet_class_name[1])
        )
    except Exception as e:
        logger.exception('Error in predict')
        raise

async def test_image_classification():
    test_image_path = os.path.join('data', 'processed', 'val', 'alb_id', '38.jpg')
    test_file = Path(test_image_path)
    with open(test_file, 'rb') as f:
        content = f.read()
    class test_file_class:
        def __init__(self, filename, content):
            self.filename = filename
            self.content = content
        async def read(self):
            return self.content
    test_file = test_file_class(filename=test_file.name, content=content)
    llm_result = await image_classification_llm(test_file, list(DOCUMENT_CLASSES.values()))
    test_file = test_file_class(filename=test_file.name, content=content)
    mobilenet_result = await image_classification_mobilenet(test_file, DOCUMENT_CLASSES)
    
    return {
        'llm_result': llm_result,
        'mobilenet_result': mobilenet_result
    }

if __name__ == '__main__':
    '''results = asyncio.run(test_image_classification())
    print(f'LLM result: {results['llm_result']}')
    print(f'MobileNet result: {results['mobilenet_result']}')'''
    
    logger.info('Starting test server on http://localhost:8000')
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')

