import os
import csv
from dotenv import load_dotenv
from openai import OpenAI
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from inference.openai_inference import image_classification_llm

DATA_DIR = "./data/processed/test"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print(f"client: {client}")
results = []
results.append(["filename", "real_class", "predicted_class", "confidence"])


def run_image_classification(img_bytes, class_list):
    return asyncio.run(image_classification_llm(img_bytes, class_list))


files = []
for subdir, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        files.append(os.path.join(subdir, filename))
for file_path in files:
    filename = os.path.basename(file_path)
    folder_name = os.path.basename(os.path.dirname(file_path))
    real_class = config.DOCUMENT_CLASSES.get(folder_name, "Unknown")
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    pred_class, confidence = run_image_classification(
        img_bytes, list(config.DOCUMENT_CLASSES.values())
    )
    results.append([filename, real_class, pred_class, confidence])
    print(f"Processed {filename}: {real_class} -> {pred_class} ({confidence})")
with open("training/llm_results.csv", "w", newline="") as f:
    csv.writer(f).writerows(results)
