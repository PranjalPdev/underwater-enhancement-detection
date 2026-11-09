import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. Paths & Parameters
# =========================
train_dir = r"classifier_data/train"
val_dir   = r"classifier_data/val"
test_dir  = r"classifier_data/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
MODEL_PATH = "efficientnet_classifier_fixed.keras"
CLASS_MAP_PATH = "class_indices.json"

# =========================
# 2. Data Generators
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, 
    class_mode="categorical", shuffle=True, seed=42
)
val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, 
    class_mode="categorical", shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, 
    class_mode="categorical", shuffle=False
)

# Save class mapping
class_indices = train_gen.class_indices
class_names = list(class_indices.keys())
with open(CLASS_MAP_PATH, "w") as f:
    json.dump(class_indices, f)

# =========================
# 3. Dynamic Class Weighting
# =========================
# Calculates physical distribution to heavily penalize missing 'low_light' or 'good'
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"\n[INFO] Computed Class Weights to prevent collapse: {class_weights_dict}\n")

# =========================
# 4. Model Architecture
# =========================
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = True

# Fine-tune only the top 20 layers to learn underwater optics instead of ImageNet objects
for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 5. Training
# =========================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)
model.save(MODEL_PATH)
print(f"\n[INFO] Model saved to: {MODEL_PATH}")

# =========================
# 6. Evaluation & Confusion Matrix
# =========================
print("\n[INFO] Evaluating on Test Set...")
test_gen.reset()
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("\n--- CLASSIFICATION REPORT (STAGE 1) ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Generate and save Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Stage 1 Classifier Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('confusion_matrix_stage1.png', dpi=300)
print("[INFO] Saved 'confusion_matrix_stage1.png' for the manuscript.")