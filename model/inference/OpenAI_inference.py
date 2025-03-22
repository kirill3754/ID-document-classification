import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

async def image_classification_llm(image: UploadFile, classes) -> str:
    try:
        image_bytes = await image.read()
        
        # Encode the image bytes to base64
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
        prompt = f"Classify the following image into one of the following classes: {', '.join(classes)}. return one class name only."
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f'Error in OPENAI inference: {str(e)}')
        return ('Error in classification', 0.0)
    
    return 'rus_internalpassport'

