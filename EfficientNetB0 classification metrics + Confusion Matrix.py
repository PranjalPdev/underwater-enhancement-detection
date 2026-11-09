# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:42:19 2026

@author: pppan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ================= USER PATHS =================
CLASSIFIER_MODEL_PATH = r"C:\internnn\efficientnet_classifier.keras"
TEST_DATA_DIR = r"C:\internnn\classifier_data\test" # Directory with subfolders: /good, /low_light, /turbid
# ==============================================

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("[INFO] Loading EfficientNetB0 Classifier...")
model = load_model(CLASSIFIER_MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Predict
print("[INFO] Evaluating Classifier...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Print Classification Report for IEEE Table
print("\n" + "="*50)
print("     CLASSIFICATION REPORT (STAGE 1)")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

# Plot and Save Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Stage 1: EfficientNetB0 Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_stage1.png', dpi=300)
print("[SUCCESS] Saved 'confusion_matrix_stage1.png' for Paper Section V.")