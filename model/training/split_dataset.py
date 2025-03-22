import os
import shutil
import random

RAW_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed"

TRAIN = 0.7
VAL = 0.15

for folder_name in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, folder_name), exist_ok=True)

for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith('.jpg')]
    random.shuffle(images)
    n = len(images)
    n_train = int(TRAIN * n)
    n_val = int(VAL * n)
    train_images = images[:n_train]
    val_images = images[n_train:n_train+n_val]
    test_images = images[n_train+n_val:]
    
    for subset_type, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        new_folder = os.path.join(OUTPUT_DIR, subset_type, class_name)
        os.makedirs(new_folder, exist_ok=True)
        for image in images:
            shutil.copy2(os.path.join(class_path, image), os.path.join(new_folder, image))

