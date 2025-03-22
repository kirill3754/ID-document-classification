import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import logging

logger = logging.getLogger(__name__)

IMG_SIZE = 224
MODEL_PATH = 'models/mobilenetv2_best.h5'
INDICES_PATH = 'models/class_indices.npy'

async def image_classification_mobilenet(image_file, class_dict):
    '''Simplified MobileNet inference function that works with class dictionary
    
    Returns:
        tuple: (prediction, confidence) where prediction is the class name and confidence is a float
    '''
    
    try:
        # Load model
        # model = load_model()
        model = keras.models.load_model(MODEL_PATH)
        
        # Read and preprocess image
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array and preprocess
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        
        # Predict
        predictions = model.predict(image_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        # Get the class key
        class_indices = np.load(INDICES_PATH, allow_pickle=True).item()
        inverted_class_indices = {v: k for k, v in class_indices.items()}
        predicted_key = inverted_class_indices.get(predicted_idx, None)
        
        # Map to the provided class dictionary description
        if predicted_key in class_dict:
            prediction = class_dict[predicted_key]
        else:
            # Fallback to the key if not found in the dictionary
            prediction = predicted_key
            
        return (prediction, confidence)
            
    except Exception as e:
        logger.error(f'Error in MobileNet inference: {str(e)}')
        return ('Error in classification', 0.0)
