import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path to import from inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.OpenAI_inference import image_classification_llm

# Define paths
DATA_DIR = '../data/processed'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load class names from test directory
TEST_DIR = os.path.join(DATA_DIR, 'test')
class_names = sorted(os.listdir(TEST_DIR))
print(f"Found {len(class_names)} classes: {class_names}")

# Check if results already exist
results_file = os.path.join(RESULTS_DIR, 'llm_results.json')

if not os.path.exists(results_file):
    # Process all test images
    print(f"Testing model on {len(class_names)} classes...")
    results = []
    total_images = 0
    
    # Count total images for progress tracking
    for class_name in class_names:
        class_dir = os.path.join(TEST_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        total_images += len(image_files)
    
    # Process each image
    processed = 0
    for class_name in class_names:
        class_dir = os.path.join(TEST_DIR, class_name)
        print(f"Processing class: {class_name}")
        
        for img_name in os.listdir(class_dir)[:5]:
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            
            processed += 1
            print(f"Image {processed}/{total_images}: {img_name}")
            
            # Read image data
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            # Get prediction
            try:
                prediction = image_classification_llm(img_data, class_names)
                results.append({
                    "image": img_path,
                    "true_class": class_name,
                    "prediction": prediction.strip()
                })
                print(f"  Prediction: {prediction.strip()}")
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
else:
    # Load existing results
    print(f"Loading existing results from {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)

# Analyze results
valid_results = [r for r in results if r['prediction'] != "Error in classification"]
print(f"Analyzing {len(valid_results)} valid results out of {len(results)} total")

# Extract true and predicted classes
y_true = [r['true_class'] for r in valid_results]
y_pred = [r['prediction'] for r in valid_results]

# Calculate per-class accuracy
per_class_accuracy = {}
for class_name in class_names:
    class_indices = [i for i, c in enumerate(y_true) if c == class_name]
    if len(class_indices) > 0:
        correct = sum(y_pred[i] == class_name for i in class_indices)
        accuracy = correct / len(class_indices)
        per_class_accuracy[class_name] = accuracy

# Print per-class accuracy
print('\nPer-class accuracy:')
for class_name, accuracy in per_class_accuracy.items():
    print(f'{class_name}: {accuracy:.4f}')

# Print classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

# Create and plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=class_names)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('OpenAI Model Results')
plt.colorbar()
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(RESULTS_DIR, 'llm_confusion_matrix.png'))
plt.show()
