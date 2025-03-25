import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import config
from loguru import logger


async def image_classification_mobilenet(image_bytes: bytes, class_dict: dict) -> tuple:
    try:
        model = keras.models.load_model(config.MODEL_PATH)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((config.IMG_SIZE, config.IMG_SIZE))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        logger.info("CNN preprocessed the image")

        predictions = model.predict(image_array)
        logger.info("CNN made predictions")
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100.0

        class_indices = np.load(config.INDICES_PATH, allow_pickle=True).item()
        inverted_class_indices = {v: k for k, v in class_indices.items()}
        predicted_key = inverted_class_indices.get(predicted_idx, None)
        if predicted_key in class_dict:
            prediction = class_dict[predicted_key]
        else:
            prediction = predicted_key
        logger.info(f"CNN prediction: {prediction}, confidence: {confidence}")
        return (prediction, confidence)

    except Exception as e:
        logger.error(f"Error in MobileNet inference: {str(e)}")
        return ("Error in classification", 0.0)
