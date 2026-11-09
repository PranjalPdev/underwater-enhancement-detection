import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import json
import hashlib
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# =====================================================================
# CONFIGURATION
# =====================================================================
TRAIN_DIR = Path(r"C:\internnn\classifier_data\train")
VAL_DIR = Path(r"C:\internnn\classifier_data\test")
RAW_VAL_IMAGES = Path(r"C:\internnn\URPC2020\images\test")

PROPOSED_OUT_DIR = Path(
    r"C:\internnn\temp_eval\proposed_pipeline"
)
MODEL_SAVE_PATH = Path(
    r"C:\internnn\efficientnet_classifier_fixed.keras"
)

FIVE_LAYER_WEIGHTS = Path(
    r"C:\internnn\checkpoints\waternet_best.pth"
)
RESIDUAL_WEIGHTS = Path(
    r"C:\internnn\checkpoints\uwcnn_best.pth"
)

CLASSES = ["good", "low_light", "turbid"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

IMG_SIZE = (224, 224)
RESTORATION_SIZE = (640, 640)

BATCH_SIZE = 32
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 10
SEED = 42
RESIDUAL_ALPHA = 0.35

# CPU restoration makes this run's execution device explicit.
# TensorFlow independently selects its available training device.
RESTORATION_DEVICE = torch.device("cpu")

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

RUN_DIR = PROPOSED_OUT_DIR.parent / (
    f"{PROPOSED_OUT_DIR.name}_{RUN_ID}"
)
OUTPUT_IMAGES = RUN_DIR / "images"

# Preserve the previous checkpoint instead of overwriting it.
RUN_MODEL_PATH = MODEL_SAVE_PATH.with_name(
    f"{MODEL_SAVE_PATH.stem}_{RUN_ID}{MODEL_SAVE_PATH.suffix}"
)

tf.keras.utils.set_random_seed(SEED)
torch.manual_seed(SEED)


# =====================================================================
# FILE AND DATASET AUDIT
# =====================================================================
def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_sha256(image):
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode())
    digest.update(str(image.dtype).encode())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def scan_images(root, labeled=False):
    if not root.is_dir():
        raise FileNotFoundError(f"Directory missing: {root}")

    records = []
    skipped = []

    sources = (
        [(label, root / label) for label in CLASSES]
        if labeled
        else [(None, root)]
    )

    for label, folder in sources:
        if not folder.is_dir():
            raise FileNotFoundError(f"Directory missing: {folder}")

        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                skipped.append({
                    "path": str(path),
                    "reason": "OpenCV could not decode image",
                })
                continue

            records.append({
                "path": str(path.resolve()),
                "label": label,
                "file_sha256": file_sha256(path),
                "pixel_sha256": decoded_sha256(image),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
            })

    return records, skipped


def remove_reference_matches(records, references, reason):
    file_hashes = {r["file_sha256"] for r in references}
    pixel_hashes = {r["pixel_sha256"] for r in references}

    kept, removed = [], []

    for record in records:
        if (
            record["file_sha256"] in file_hashes
            or record["pixel_sha256"] in pixel_hashes
        ):
            removed.append({**record, "reason": reason})
        else:
            kept.append(record)

    return kept, removed


def deduplicate_labeled(records, split_name):
    seen_pixels = {}
    kept, removed = [], []

    for record in records:
        key = record["pixel_sha256"]

        if key not in seen_pixels:
            seen_pixels[key] = record
            kept.append(record)
            continue

        previous = seen_pixels[key]
        if previous["label"] != record["label"]:
            raise ValueError(
                f"Conflicting labels in {split_name}:\n"
                f"{previous['path']}\n{record['path']}"
            )

        removed.append({
            **record,
            "reason": f"Duplicate within {split_name}",
        })

    return kept, removed


def assert_all_classes(records, split_name):
    counts = Counter(r["label"] for r in records)

    missing = [name for name in CLASSES if counts[name] == 0]
    if missing:
        raise ValueError(
            f"{split_name} has no valid images for: {missing}"
        )

    return {name: counts[name] for name in CLASSES}


def prepare_manifests():
    train, train_bad = scan_images(TRAIN_DIR, labeled=True)
    validation, val_bad = scan_images(VAL_DIR, labeled=True)
    detection, det_bad = scan_images(RAW_VAL_IMAGES)

    write_json(
        RUN_DIR / "unreadable_images.json",
        train_bad + val_bad + det_bad,
    )

    # Do not silently reduce the detection manifest.
    if det_bad:
        raise RuntimeError(
            "Unreadable detection images found. "
            "See unreadable_images.json."
        )
    if not detection:
        raise RuntimeError("No detection images found.")

    removed = []

    validation, rejected = remove_reference_matches(
        validation,
        detection,
        "Validation image matches detection manifest",
    )
    removed.extend(rejected)

    validation, rejected = deduplicate_labeled(
        validation, "validation"
    )
    removed.extend(rejected)

    train, rejected = remove_reference_matches(
        train,
        detection + validation,
        "Training image matches detection or validation",
    )
    removed.extend(rejected)

    train, rejected = deduplicate_labeled(train, "training")
    removed.extend(rejected)

    train_counts = assert_all_classes(train, "Training")
    val_counts = assert_all_classes(validation, "Validation")

    write_json(RUN_DIR / "training_manifest.json", train)
    write_json(RUN_DIR / "validation_manifest.json", validation)
    write_json(RUN_DIR / "detection_manifest.json", detection)
    write_json(RUN_DIR / "excluded_images.json", removed)

    counts = {
        "training": train_counts,
        "validation": val_counts,
        "training_total": len(train),
        "validation_total": len(validation),
        "detection_total": len(detection),
    }
    write_json(RUN_DIR / "dataset_counts.json", counts)

    print("\nACTUAL RETAINED DATASET COUNTS")
    print(json.dumps(counts, indent=2))
    print(f"Excluded duplicate/overlap records: {len(removed)}")

    return train, validation, detection


# =====================================================================
# EFFICIENTNET TRAINING
# =====================================================================
def make_generator(records, training):
    frame = pd.DataFrame({
        "filename": [r["path"] for r in records],
        "label": [r["label"] for r in records],
    })

    if training:
        generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        generator = ImageDataGenerator()

    # No rescale=1/255: EfficientNet includes normalization.
    return generator.flow_from_dataframe(
        dataframe=frame,
        x_col="filename",
        y_col="label",
        classes=CLASSES,
        target_size=IMG_SIZE,
        color_mode="rgb",
        interpolation="bilinear",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=training,
        seed=SEED,
    )


def train_classifier(train_records, val_records):
    train_gen = make_generator(train_records, training=True)
    val_gen = make_generator(val_records, training=False)

    assert train_gen.class_indices == val_gen.class_indices

    y_train = np.asarray(train_gen.classes)
    class_ids = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=y_train,
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in zip(class_ids, weights)
    }

    write_json(
        RUN_DIR / "class_mapping.json",
        train_gen.class_indices,
    )
    write_json(RUN_DIR / "class_weights.json", class_weights)

    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(CLASSES), activation="softmax")(x)

    model = Model(inputs, outputs)

    # Reuse this callback so the best checkpoint is selected
    # across both training stages using validation loss.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(RUN_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    def callbacks(log_name):
        return [
            checkpoint,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.CSVLogger(
                str(RUN_DIR / log_name)
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

    print("\n[TRAINING] Classification head")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks("head_history.csv"),
    )

    # Start fine-tuning from the best saved head-stage weights.
    model.load_weights(str(RUN_MODEL_PATH))

    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    print("\n[TRAINING] Backbone fine-tuning")
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks("fine_tune_history.csv"),
    )

    model = tf.keras.models.load_model(
        str(RUN_MODEL_PATH), compile=False
    )

    val_gen.reset()
    probabilities = model.predict(val_gen, verbose=1)
    predictions = probabilities.argmax(axis=1)
    truth = np.asarray(val_gen.classes)

    report = classification_report(
        truth,
        predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    print("\nVALIDATION REPORT\n", report)

    (RUN_DIR / "validation_report.txt").write_text(
        report, encoding="utf-8"
    )

    matrix = confusion_matrix(
        truth,
        predictions,
        labels=list(range(len(CLASSES))),
    )
    pd.DataFrame(
        matrix,
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    ).to_csv(RUN_DIR / "confusion_matrix.csv")

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(
        RUN_DIR / "confusion_matrix_efficientnet.png",
        dpi=300,
    )
    plt.close()

    # Release the training model before loading PyTorch branches.
    del model, base, train_gen, val_gen
    tf.keras.backend.clear_session()


# =====================================================================
# RETAINED ENHANCEMENT ARCHITECTURES
# =====================================================================
class FiveLayerEnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DenseResidualEnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Attribute names preserve the supplied checkpoint keys.
        input_channels = [3, 16, 16, 48, 64, 80, 96, 112, 128]
        for index, channels in enumerate(input_channels, start=1):
            setattr(
                self,
                f"conv{index}",
                nn.Sequential(
                    nn.Conv2d(channels, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                ),
            )
        self.conv10 = nn.Conv2d(144, 3, 3, padding=1)

    def forward(self, x, alpha=RESIDUAL_ALPHA):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        features = torch.cat([x1, x2, x3], dim=1)
        for index in range(4, 10):
            new_features = getattr(self, f"conv{index}")(features)
            features = torch.cat([features, new_features], dim=1)

        residual = self.conv10(features)
        return torch.clamp(x + alpha * residual, 0.0, 1.0)


def load_enhancer(model, checkpoint_path):
    state = torch.load(
        checkpoint_path,
        map_location=RESTORATION_DEVICE,
        weights_only=True,
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # No partial loading: incompatible checkpoints must fail.
    model.load_state_dict(state, strict=True)
    return model.to(RESTORATION_DEVICE).eval()


@torch.inference_mode()
def restore_image(model, bgr_image):
    height, width = bgr_image.shape[:2]

    resized = cv2.resize(
        bgr_image,
        RESTORATION_SIZE,
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0

    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(RESTORATION_DEVICE)
    )

    output = model(tensor)

    if not torch.isfinite(output).all().item():
        raise RuntimeError("Enhancement produced NaN or infinity.")

    if tuple(output.shape) != tuple(tensor.shape):
        raise RuntimeError(
            f"Unexpected output shape: {tuple(output.shape)}"
        )

    output = (
        output.clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    rgb_uint8 = np.rint(output * 255.0).astype(np.uint8)
    bgr_output = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    return cv2.resize(
        bgr_output,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


# =====================================================================
# ARGMAX ROUTING AND OUTPUT GENERATION
# =====================================================================
def generate_pipeline_outputs(detection_records):
    classifier = tf.keras.models.load_model(
        str(RUN_MODEL_PATH), compile=False
    )
    class_indices = json.loads(
        (RUN_DIR / "class_mapping.json").read_text()
    )
    index_to_class = {
        int(index): name
        for name, index in class_indices.items()
    }

    five_layer = load_enhancer(
        FiveLayerEnhancementCNN(), FIVE_LAYER_WEIGHTS
    )
    residual = load_enhancer(
        DenseResidualEnhancementCNN(), RESIDUAL_WEIGHTS
    )

    counts = Counter()
    route_records = []

    for record in tqdm(detection_records, desc="Routing"):
        source = Path(record["path"])
        destination = OUTPUT_IMAGES / source.name

        # Match the RGB and bilinear preprocessing used in training.
        pil_image = tf.keras.utils.load_img(
            source,
            color_mode="rgb",
            target_size=IMG_SIZE,
            interpolation="bilinear",
        )
        classifier_input = tf.keras.utils.img_to_array(pil_image)
        pil_image.close()

        probabilities = classifier(
            classifier_input[None, ...],
            training=False,
        ).numpy()[0]

        predicted_index = int(np.argmax(probabilities))
        route = index_to_class[predicted_index]

        if route == "good":
            shutil.copy2(source, destination)
        else:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot decode: {source}")

            enhancer = (
                five_layer if route == "low_light" else residual
            )
            processed = restore_image(enhancer, image)

            if not cv2.imwrite(str(destination), processed):
                raise IOError(f"Failed to save: {destination}")

        counts[route] += 1

        route_records.append({
            "source": str(source),
            "output": str(destination),
            "route": route,
            **{
                f"p_{name}": float(probabilities[index])
                for index, name in index_to_class.items()
            },
        })

    pd.DataFrame(route_records).to_csv(
        RUN_DIR / "routing_manifest.csv",
        index=False,
    )
    write_json(
        RUN_DIR / "routing_counts.json",
        {name: counts[name] for name in CLASSES},
    )

    assert sum(counts.values()) == len(detection_records)

    print("\nROUTING COUNTS")
    print({name: counts[name] for name in CLASSES})


# =====================================================================
# MAIN
# =====================================================================
def main():
    for checkpoint in [FIVE_LAYER_WEIGHTS, RESIDUAL_WEIGHTS]:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_IMAGES.mkdir()
    RUN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_json(RUN_DIR / "configuration.json", {
        "router": "EfficientNetB0",
        "routing_policy": "argmax",
        "classes": CLASSES,
        "seed": SEED,
        "classifier_input": [224, 224],
        "classifier_pixel_range": [0, 255],
        "restoration_input": [640, 640],
        "residual_alpha": RESIDUAL_ALPHA,
        "restoration_device": str(RESTORATION_DEVICE),
        "model_output": str(RUN_MODEL_PATH),
        "tensorflow_version": tf.__version__,
        "torch_version": torch.__version__,
        "five_layer_checkpoint_sha256": file_sha256(
            FIVE_LAYER_WEIGHTS
        ),
        "residual_checkpoint_sha256": file_sha256(
            RESIDUAL_WEIGHTS
        ),
        "duplicate_audit": (
            "File SHA-256 and exact decoded-pixel SHA-256; "
            "not a near-duplicate or scene-level audit"
        ),
    })

    train, validation, detection = prepare_manifests()
    train_classifier(train, validation)
    generate_pipeline_outputs(detection)

    print("\nCOMPLETED")
    print("Model:", RUN_MODEL_PATH)
    print("Run records:", RUN_DIR)
    print("Processed images:", OUTPUT_IMAGES)
    print(
        "These are EfficientNet argmax outputs. "
        "YOLO evaluation and timing have not been performed."
    )


if __name__ == "__main__":
    main()