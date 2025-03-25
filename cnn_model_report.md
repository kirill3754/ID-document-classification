# MobileNetV2 CNN Model Report

## Architecture
- Frozen base model MobileNetV2 pretrained on ImageNet
- Input: 224 × 224 × 3
- GlobalAveragePooling2D
- Classification Head:
  - Dense layer (128 units, ReLU activation)
  - Dropout (0.5)
  - Output layer (10 units, Softmax activation)

## Training Configuration
- **Dataset**: Processed images in train/val/test directories 700/150/150
- **Image Size**: 224 × 224 pixels
- **Batch Size**: 35
- **Preprocessing**: scaling pixels from [0, 255] to [-1, 1] and image resizing
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Epochs**: 20
- **Callbacks**:
  - Early Stopping (patience=5, monitor='val_loss')
  - Model Checkpoint (monitor='val_accuracy', save best only)

## Training Process
- Transfer Learning Approach:
  - Frozen MobileNetV2 base for feature extraction
  - A classification head trained from scratch

## Results

### Training & Validation Loss
![alt text](image.png)
- Training loss: 0.0514
- Training accuracy: 0.9957
- Val loss: 0.6409
- Val accuracy: 0.7867


### Test Data Classification Report
- Test Loss: 0.58
- Test Accuracy: 0.79

```
                      precision    recall  f1-score   support

              alb_id       0.62      0.67      0.65        15
        aze_passport       0.88      0.93      0.90        15
              esp_id       0.67      0.93      0.78        15
              est_id       0.81      0.87      0.84        15
              fin_id       0.75      0.20      0.32        15
        grc_passport       0.86      0.80      0.83        15
        lva_passport       0.87      0.87      0.87        15
rus_internalpassport       1.00      0.80      0.89        15
        srb_passport       0.70      0.93      0.80        15
              svk_id       0.81      0.87      0.84        15

            accuracy                           0.79       150
           macro avg       0.80      0.79      0.77       150
        weighted avg       0.80      0.79      0.77       150
```

### Confusion Matrix
```
[[10  0  3  0  0  0  0  0  1  1]
 [ 0 14  0  1  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  0  0  1]
 [ 0  0  0 13  1  0  0  0  1  0]
 [ 5  1  3  2  3  0  0  0  0  1]
 [ 0  0  0  0  0 12  2  0  1  0]
 [ 0  1  0  0  0  0 13  0  1  0]
 [ 0  0  0  0  0  1  0 12  2  0]
 [ 0  0  0  0  0  1  0  0 14  0]
 [ 1  0  1  0  0  0  0  0  0 13]]
```

## Model Save
- Best model saved at: `../models/mobilenetv2_best.h5`
- Class indices saved at: `../models/class_indices.npy`
