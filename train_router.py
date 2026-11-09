# # step1_train_router.py
# import json
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from common_feature import extract_optical_features

# TRAIN_DIR = Path(r"classifier_data/train")
# TEST_DIR  = Path(r"classifier_data/test")
# MODEL_OUT = Path("optical_rf_stage1.joblib")
# META_OUT  = Path("router_metadata.json")

# print("[INFO] Indexing training and evaluation datasets...")
# train_gen = ImageDataGenerator().flow_from_directory(
#     TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False
# )
# test_gen = ImageDataGenerator().flow_from_directory(
#     TEST_DIR, target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False
# )

# class_indices = train_gen.class_indices
# idx_to_class = {v: k for k, v in class_indices.items()}

# print(f"[INFO] Extracting features for {len(train_gen.filepaths)} training images...")
# X_train, y_train = [], []
# for path, label in zip(train_gen.filepaths, train_gen.classes):
#     feat = extract_optical_features(path)
#     if feat is not None:
#         X_train.append(feat)
#         y_train.append(label)

# X_train, y_train = np.array(X_train), np.array(y_train)

# # Fit model with explicit balanced weights
# clf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=6,
#     random_state=42,
#     class_weight="balanced",
#     n_jobs=-1
# )
# clf.fit(X_train, y_train)

# print(f"[INFO] Evaluating model on {len(test_gen.filepaths)} validation images...")
# X_test, y_test = [], []
# for path, label in zip(test_gen.filepaths, test_gen.classes):
#     feat = extract_optical_features(path)
#     if feat is not None:
#         X_test.append(feat)
#         y_test.append(label)

# X_test, y_test = np.array(X_test), np.array(y_test)
# y_pred = clf.predict(X_test)

# # Report actual metrics
# class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
# print("\n" + "="*60)
# print("ACTUAL STAGE 1 RANDOM FOREST CLASSIFICATION REPORT")
# print("="*60)
# report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
# print(report)

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.title("Stage 1: Verified RF Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("Ground Truth")
# plt.tight_layout()
# plt.savefig("confusion_matrix_stage1_rf.png", dpi=300)
# print("[INFO] Saved authentic confusion matrix to 'confusion_matrix_stage1_rf.png'")

# # Serialize model & metadata
# joblib.dump(clf, MODEL_OUT)
# with open(META_OUT, "w") as f:
#     json.dump({
#         "class_indices": class_indices,
#         "idx_to_class": idx_to_class,
#         "features": [
#             "laplacian_var", "l_mean", "l_std", "a_mean", "b_mean", 
#             "saturation", "contrast", "dark_prior", "attenuation_r"
#         ]
#     }, f, indent=4)
# print(f"[SUCCESS] Saved router estimator to '{MODEL_OUT}' and metadata to '{META_OUT}'\n")
# from collections import Counter

# print("Actual y_train length:", len(y_train))
# print("Actual X_train length:", len(X_train))

# counts = Counter(int(label) for label in y_train)
# index_to_class = {
#     index: name
#     for name, index in train_gen.class_indices.items()
# }

# print("\nActual training counts:")
# for index, name in sorted(index_to_class.items()):
#     print(f"{name}: {counts.get(index, 0)}")

# print("\nCount total:", sum(counts.values()))

# assert len(X_train) == len(y_train), "Feature/label count mismatch".
# verify_exact_training_split.py
# from pathlib import Path
# from collections import Counter
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from common_feature import extract_optical_features

# TRAIN_DIR = Path(r"classifier_data/train")

# # 1. Check raw filesystem files vs. Keras ImageDataGenerator
# all_files = {d.name: list(d.glob("*.*")) for d in TRAIN_DIR.iterdir() if d.is_dir()}
# print("Raw filesystem counts (*.*):", {k: len(v) for k, v in all_files.items()})

# train_gen = ImageDataGenerator().flow_from_directory(
#     TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False
# )

# # Find the exact file(s) present on disk but ignored by flow_from_directory
# gen_set = set(Path(p).resolve() for p in train_gen.filepaths)
# for cls_name, files in all_files.items():
#     for f in files:
#         if f.resolve() not in gen_set:
#             print(f"[DISCREPANCY FILE] Ignored by Keras in '{cls_name}': {f.name}")

# # 2. Check exact valid feature vectors passed to clf.fit(X_train, y_train)
# y_train = []
# skipped_imread = []
# for path, label in zip(train_gen.filepaths, train_gen.classes):
#     feat = extract_optical_features(path)
#     if feat is not None:
#         y_train.append(label)
#     else:
#         skipped_imread.append(path)

# idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
# counts_by_name = {idx_to_class[k]: v for k, v in Counter(y_train).items()}

# print("\n--- EXACT RECORD FOR MANUSCRIPT ---")
# print("Valid training samples passed to clf.fit:", len(y_train))
# print("Training counts by label:", counts_by_name)
# print("Class mapping:", train_gen.class_indices)
# if skipped_imread:
#     print("Files skipped due to unreadable cv2.imread:", skipped_imread).
import cv2
import numpy as np

def extract_optical_features_from_bgr(img_bgr: np.ndarray) -> list:
    """
    Extracts 9 optics-inspired features from an in-memory BGR image.
    Resizes to 224x224 to standardize scale-dependent derivatives.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid image array passed to feature extractor.")
        
    img_resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    hsv  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # 1. Laplacian blur variance (high-frequency edge energy)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. L* luminance mean (exposure level)
    l_mean = float(lab[:, :, 0].mean())
    
    # 3. L* luminance standard deviation (dynamic range)
    l_std = float(lab[:, :, 0].std())
    
    # 4. a* chromaticity mean (green-red axis)
    a_mean = float(lab[:, :, 1].mean())
    
    # 5. b* chromaticity mean (blue-yellow axis)
    b_mean = float(lab[:, :, 2].mean())
    
    # 6. HSV saturation mean (color concentration)
    saturation = float(hsv[:, :, 1].mean())
    
    # 7. Gray RMS contrast (overall intensity dispersion)
    contrast = float(gray.std())
    
    # 8. Dark channel proxy (mean across minimum color channels)
    dark_prior = float(np.min(img_resized, axis=2).mean())
    
    # 9. Red/Blue attenuation ratio (safeguarded against division by zero)
    attenuation_r = float(img_resized[:, :, 2].mean() / (img_resized[:, :, 0].mean() + 1e-5))
    
    return [
        laplacian_var, l_mean, l_std, a_mean, b_mean, 
        saturation, contrast, dark_prior, attenuation_r
    ]

def extract_optical_features(img_path: str) -> list:
    img = cv2.imread(img_path)
    if img is None:
        return None
    return extract_optical_features_from_bgr(img)