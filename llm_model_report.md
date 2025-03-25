# GPT-4o-mini LLM Model Report

## Architecture
- Zero-shot OpenAI GPT-4o-mini model
- Multimodal model input

## Configuration
- **Model**: GPT-4o-mini
- **Temperature**: 0.0 (deterministic response)
- **Prompt**: "Classify the following image into one of the following classes: {classes}. return one class name only."
- **Output Confidence**: Calculated from token logprobs

## Results

### Classification Report
- Test Accuracy: 0.96

```
                             precision    recall  f1-score   support

         ID Card of Albania       1.00      1.00      1.00        15
         ID Card of Estonia       0.83      1.00      0.91        15
         ID Card of Finland       1.00      1.00      1.00        15
        ID Card of Slovakia       1.00      1.00      1.00        15
           ID Card of Spain       1.00      1.00      1.00        15
Internal passport of Russia       1.00      0.73      0.85        15
     Passport of Azerbaijan       1.00      0.93      0.97        15
         Passport of Greece       1.00      1.00      1.00        15
         Passport of Latvia       1.00      0.93      0.97        15
         Passport of Serbia       1.00      1.00      1.00        15
                      other       0.00      0.00      0.00         0

                   accuracy                           0.96       150
                  macro avg       0.89      0.87      0.88       150
               weighted avg       0.98      0.96      0.97       150
```

### Confusion Matrix

```
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0 15  0  0  0  0  0  0]
 [ 0  3  0  0  0 11  0  0  0  0  1]
 [ 0  0  0  0  0  0 14  0  0  0  1]
 [ 0  0  0  0  0  0  0 15  0  0  0]
 [ 0  0  0  0  0  0  0  0 14  0  1]
 [ 0  0  0  0  0  0  0  0  0 15  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]
```

## Cost
0.8 USD per 150 test samples
