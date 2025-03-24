# MobileNetV2 Model Report

## Model Architecture
- Base: MobileNetV2 (pretrained on ImageNet)
- Input Shape: 224 × 224 × 3
- Feature Extraction: Frozen base model + GlobalAveragePooling2D
- Classification Head:
  - Dense layer (128 units, ReLU activation)
  - Dropout (0.5)
  - Output layer (10 units, Softmax activation)

## Training Configuration
- **Dataset**: Processed images in train/val/test directories
- **Image Size**: 224 × 224 pixels
- **Batch Size**: 35
- **Preprocessing**: MobileNetV2 standard preprocessing
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Epochs**: 20 (with early stopping)
- **Callbacks**:
  - Early Stopping (patience=5, monitor='val_loss')
  - Model Checkpoint (monitor='val_accuracy', save best only)

## Training Process
- Transfer Learning Approach:
  - Frozen MobileNetV2 base for feature extraction
  - Custom classification head trained from scratch
- Data Augmentation: None additional beyond preprocessing

## Results

### Performance Metrics
- Test Accuracy: [Final accuracy value]
- Test Loss: [Final loss value]

### Per-Class Performance
| Class | Accuracy |
|-------|----------|
| [class 1] | [accuracy] |
| [class 2] | [accuracy] |
| ... | ... |

### Classification Report
```
[Classification report content]
```

## Visualizations

### Training & Validation Loss
![Training and Validation Loss curves]

### Confusion Matrix
```
[Confusion matrix content]
```

## Model Artifacts
- Best model saved at: `../models/mobilenetv2_best.h5`
- Final model saved at: `../models/mobilenetv2_final.h5`
- Training history saved at: `../models/history.pkl`
- Class indices saved at: `../models/class_indices.npy`
