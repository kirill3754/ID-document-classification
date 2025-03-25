import base64
import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
from loguru import logger
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client created")
except:
    client = None
    logger.error("OpenAI client is disabled")


async def image_classification_llm(image_bytes, classes) -> str:
    if client is None:
        return "OpenAI is disabled", 0.0
    try:
        # Encode the image bytes to base64
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = f"Classify the following image into one of the following classes: {', '.join(classes)}. return one class name only."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ]

        params = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.0,
            "logprobs": True,
        }
        logger.info("Sending image to OpenAI")
        completion = await client.chat.completions.create(**params)
        linprob = min(
            [
                np.round(np.exp(logprob.logprob) * 100, 2)
                for logprob in completion.choices[0].logprobs.content
            ]
        )
        logger.info(f"OpenAI response: {completion.choices[0].message.content}")
        return completion.choices[0].message.content, linprob
    except Exception as e:
        logger.error(f"Error in OPENAI inference: {str(e)}")
        return ("Error in classification", 0.0)

    return "test_class"
