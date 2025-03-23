import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import config
from loguru import logger


async def image_classification_mobilenet(image_bytes: bytes, class_dict: dict) -> tuple:
    try:
        # Load model
        model = keras.models.load_model(config.MODEL_PATH)

        # Read and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((config.IMG_SIZE, config.IMG_SIZE))

        # Convert to array and preprocess
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

        # Predict
        predictions = model.predict(image_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100.0

        # Get the class key
        class_indices = np.load(config.INDICES_PATH, allow_pickle=True).item()
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
        logger.error(f"Error in MobileNet inference: {str(e)}")
        return ("Error in classification", 0.0)
