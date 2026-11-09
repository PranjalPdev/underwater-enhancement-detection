import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ultralytics
from ultralytics import YOLO
from tqdm import tqdm


# =====================================================================
# CONFIGURATION
# =====================================================================
RUN_DIR = Path(
    r"C:\internnn\temp_eval"
    r"\proposed_pipeline_20261019T203221_524771Z"
)

RAW_IMAGES_DIR = Path(r"C:\internnn\URPC2020\images\test")
RAW_LABELS_DIR = RUN_DIR / "labels_cleaned_zero_height_v1"

YOLO_WEIGHTS = Path(
    r"C:\Users\pppan\runs\detect"
    r"\urpc2020_yolov8n-9\weights\best.pt"
)

FIVE_LAYER_WEIGHTS = Path(
    r"C:\internnn\checkpoints\waternet_best.pth"
)
RESIDUAL_WEIGHTS = Path(
    r"C:\internnn\checkpoints\uwcnn_best.pth"
)

DETECTION_MANIFEST = RUN_DIR / "detection_manifest.json"
ROUTING_MANIFEST = RUN_DIR / "routing_manifest.csv"

EXPECTED_IMAGES = 800
CLASS_NAMES = ["holothurian", "echinus", "scallop", "starfish"]

DEVICE = torch.device("cpu")
BATCH_SIZE = 16
IMAGE_SIZE = 640
ALPHA = 0.35

# Additional 6 detector evaluations; enable if desired.
RUN_ROUTEWISE = False

# Missing labels cause an error rather than silently becoming backgrounds.
# For genuine backgrounds, supply an explicitly empty label file.
REQUIRE_LABEL_FILES = True

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
EVAL_DIR = RUN_DIR / f"detection_evaluation_{RUN_ID}"


# =====================================================================
# UTILITIES
# =====================================================================
def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write: {path}")


def validate_label(path):
    if not path.exists():
        if REQUIRE_LABEL_FILES:
            raise FileNotFoundError(
                f"Missing label: {path}\n"
                "Create an empty file only if this is a verified background."
            )
        return

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{path}, line {line_number}: expected class x y w h"
            )

        values = np.asarray(fields, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite annotation: {path}")

        class_id = values[0]
        x, y, w, h = values[1:]

        if class_id != int(class_id) or not 0 <= class_id < 4:
            raise ValueError(f"Invalid class ID: {path}")

        if not (
            0 <= x <= 1 and 0 <= y <= 1
            and 0 < w <= 1 and 0 < h <= 1
        ):
            raise ValueError(f"Invalid normalized box: {path}")


# =====================================================================
# EXACT SUPPLIED RESTORATION ARCHITECTURES
# =====================================================================
class FiveLayerEnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DenseResidualEnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3, 16, 16, 48, 64, 80, 96, 112, 128]

        for i, in_channels in enumerate(channels, start=1):
            setattr(
                self,
                f"conv{i}",
                nn.Sequential(
                    nn.Conv2d(in_channels, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                ),
            )

        self.conv10 = nn.Conv2d(144, 3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        features = torch.cat([x1, x2, x3], dim=1)
        for i in range(4, 10):
            out = getattr(self, f"conv{i}")(features)
            features = torch.cat([features, out], dim=1)

        residual = self.conv10(features)
        return torch.clamp(x + ALPHA * residual, 0, 1)


def load_enhancer(model, path):
    state = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


@torch.inference_mode()
def enhance(model, image):
    h, w = image.shape[:2]

    resized = cv2.resize(
        image, (640, 640), interpolation=cv2.INTER_LINEAR
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0

    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )
    output = model(tensor)

    if output.shape != tensor.shape:
        raise RuntimeError("Unexpected enhancement output shape.")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Enhancement output contains NaN/infinity.")

    array = (
        output.clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    array = np.rint(array * 255.0).astype(np.uint8)
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    return cv2.resize(
        array, (w, h), interpolation=cv2.INTER_LINEAR
    )


def resize_only(image):
    h, w = image.shape[:2]
    image = cv2.resize(
        image, (640, 640), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(
        image, (w, h), interpolation=cv2.INTER_LINEAR
    )


# =====================================================================
# MANIFEST AND CHECKPOINT CHECKS
# =====================================================================
def load_records():
    source_records = json.loads(
        DETECTION_MANIFEST.read_text(encoding="utf-8")
    )
    routes = pd.read_csv(
        ROUTING_MANIFEST, keep_default_na=False
    ).to_dict("records")

    def path_key(value):
        return str(Path(value).resolve()).casefold()

    route_lookup = {}
    for row in routes:
        key = path_key(row["source"])
        if key in route_lookup:
            raise ValueError("Duplicate source in routing manifest.")
        route_lookup[key] = row

    source_keys = {path_key(r["path"]) for r in source_records}
    if source_keys != set(route_lookup):
        raise ValueError(
            "Detection and routing manifests contain different images."
        )

    if len(source_records) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(source_records)}."
        )

    names, stems = set(), set()
    records = []

    for record in source_records:
        source = Path(record["path"])
        route = route_lookup[path_key(source)]
        output = Path(route["output"])

        if source.parent.resolve() != RAW_IMAGES_DIR.resolve():
            raise ValueError(f"Unexpected source directory: {source}")

        if source.name.casefold() in names:
            raise ValueError("Duplicate output filename.")
        if source.stem.casefold() in stems:
            raise ValueError("Duplicate label stem.")

        names.add(source.name.casefold())
        stems.add(source.stem.casefold())

        if not source.is_file() or not output.is_file():
            raise FileNotFoundError(f"Missing source/output: {source}")

        if sha256_file(source) != record["file_sha256"]:
            raise RuntimeError(f"Source changed since routing: {source}")

        if route["route"] not in {"good", "low_light", "turbid"}:
            raise ValueError(f"Invalid route: {route['route']}")

        original = read_image(source)
        processed = read_image(output)

        if original.shape != processed.shape:
            raise ValueError(f"Output dimensions differ: {output}")

        if route["route"] == "good":
            if sha256_file(source) != sha256_file(output):
                raise RuntimeError(
                    f"Bypass is not byte-identical: {output}"
                )

        label = RAW_LABELS_DIR / f"{source.stem}.txt"
        validate_label(label)

        records.append({
            "source": source,
            "dynamic": output,
            "label": label,
            "route": route["route"],
        })

    return records


def check_checkpoint_identity():
    config_path = RUN_DIR / "configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for key, path in [
        ("five_layer_checkpoint_sha256", FIVE_LAYER_WEIGHTS),
        ("residual_checkpoint_sha256", RESIDUAL_WEIGHTS),
    ]:
        if sha256_file(path) != config[key]:
            raise RuntimeError(
                f"Checkpoint differs from routing run: {path}"
            )

    if float(config["residual_alpha"]) != ALPHA:
        raise RuntimeError("Residual alpha differs from routing run.")


# =====================================================================
# PREPARE ISOLATED DATASETS
# =====================================================================
def make_dataset(name):
    root = EVAL_DIR / "datasets" / name
    (root / "images").mkdir(parents=True, exist_ok=False)
    (root / "labels").mkdir()
    return root


def copy_label(record, dataset):
    if record["label"].exists():
        shutil.copy2(
            record["label"],
            dataset / "labels" / record["label"].name,
        )
    else:
        (dataset / "labels" / f"{record['source'].stem}.txt").touch()


def prepare_datasets(records):
    datasets = {
        name: make_dataset(name)
        for name in [
            "raw",
            "fixed_five_layer",
            "fixed_residual",
            "dynamic_efficientnet",
            "resize_only",
        ]
    }

    five_layer = load_enhancer(
        FiveLayerEnhancementCNN(), FIVE_LAYER_WEIGHTS
    )
    residual = load_enhancer(
        DenseResidualEnhancementCNN(), RESIDUAL_WEIGHTS
    )

    for record in tqdm(records, desc="Preparing comparison datasets"):
        source = record["source"]
        image = read_image(source)

        shutil.copy2(
            source, datasets["raw"] / "images" / source.name
        )
        shutil.copy2(
            record["dynamic"],
            datasets["dynamic_efficientnet"] / "images" / source.name,
        )

        write_image(
            datasets["fixed_five_layer"] / "images" / source.name,
            enhance(five_layer, image),
        )
        write_image(
            datasets["fixed_residual"] / "images" / source.name,
            enhance(residual, image),
        )
        write_image(
            datasets["resize_only"] / "images" / source.name,
            resize_only(image),
        )

        for dataset in datasets.values():
            copy_label(record, dataset)

    # Confirm selected fixed branches reproduce saved dynamic pixels.
    for record in tqdm(records, desc="Checking dynamic branch identity"):
        if record["route"] == "good":
            continue

        fixed_name = (
            "fixed_five_layer"
            if record["route"] == "low_light"
            else "fixed_residual"
        )
        fixed_image = read_image(
            datasets[fixed_name] / "images" / record["source"].name
        )
        dynamic_image = read_image(record["dynamic"])

        if not np.array_equal(fixed_image, dynamic_image):
            raise RuntimeError(
                "Regenerated fixed branch does not match the saved "
                f"dynamic output: {record['source'].name}. "
                "Do not evaluate mismatched implementations."
            )

    return datasets


def dataset_yaml(dataset, name):
    # JSON syntax is valid YAML and safely quotes Windows paths.
    path = EVAL_DIR / f"{name}.yaml"
    save_json(path, {
        "path": dataset.resolve().as_posix(),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": {i: c for i, c in enumerate(CLASS_NAMES)},
    })
    return path


# =====================================================================
# DETECTION EVALUATION
# =====================================================================
def evaluate(dataset, name):
    model = YOLO(str(YOLO_WEIGHTS))

    actual_names = [
        model.names[i] for i in range(len(model.names))
    ]
    if actual_names != CLASS_NAMES:
        raise RuntimeError(
            "Detector class mapping differs.\n"
            f"Expected: {CLASS_NAMES}\nActual: {actual_names}\n"
            "Verify class IDs before proceeding."
        )

    metrics = model.val(
        data=str(dataset_yaml(dataset, name)),
        split="test",
        imgsz=IMAGE_SIZE,
        rect=False,
        batch=BATCH_SIZE,
        device="cpu",
        workers=0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        half=False,
        augment=False,
        plots=True,
        save_json=True,
        project=str(EVAL_DIR / "ultralytics_results"),
        name=name,
        exist_ok=False,
        verbose=True,
    )

    present_classes = np.asarray(
        metrics.box.ap_class_index, dtype=int
    )
    class_rows = []
    for position, class_id in enumerate(present_classes):
        class_rows.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "map50": float(metrics.box.ap50[position]),
            "map50_95": float(metrics.box.ap[position]),
        })

    pd.DataFrame(class_rows).to_csv(
        EVAL_DIR / f"{name}_per_class.csv", index=False
    )

    return {
        "configuration": name,
        "images": len(list((dataset / "images").iterdir())),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def evaluate_routes(records, datasets):
    rows = []

    for route in ["good", "low_light", "turbid"]:
        selected = [r for r in records if r["route"] == route]
        if not selected:
            continue

        scores = {}
        for configuration in ["raw", "dynamic_efficientnet"]:
            name = f"route_{route}_{configuration}"
            subset = make_dataset(name)

            for record in selected:
                filename = record["source"].name
                shutil.copy2(
                    datasets[configuration] / "images" / filename,
                    subset / "images" / filename,
                )
                copy_label(record, subset)

            scores[configuration] = evaluate(subset, name)

        raw_score = scores["raw"]["map50"]
        dynamic_score = scores["dynamic_efficientnet"]["map50"]

        rows.append({
            "route": route,
            "images": len(selected),
            "raw_map50": raw_score,
            "pipeline_map50": dynamic_score,
            "delta": dynamic_score - raw_score,
        })

        pd.DataFrame(rows).to_csv(
            EVAL_DIR / "routewise_results.csv", index=False
        )

    return rows


# =====================================================================
# MAIN
# =====================================================================
def main():
    for path in [
        DETECTION_MANIFEST,
        ROUTING_MANIFEST,
        YOLO_WEIGHTS,
        FIVE_LAYER_WEIGHTS,
        RESIDUAL_WEIGHTS,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if not RAW_LABELS_DIR.is_dir():
        raise FileNotFoundError(RAW_LABELS_DIR)

    EVAL_DIR.mkdir(parents=True, exist_ok=False)

    check_checkpoint_identity()
    records = load_records()

    save_json(EVAL_DIR / "evaluation_configuration.json", {
        "source_run": str(RUN_DIR),
        "images": len(records),
        "yolo_checkpoint_sha256": sha256_file(YOLO_WEIGHTS),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "device": "cpu",
        "imgsz": IMAGE_SIZE,
        "rect": False,
        "batch": BATCH_SIZE,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "residual_alpha": ALPHA,
        "route_counts": {
            route: sum(r["route"] == route for r in records)
            for route in ["good", "low_light", "turbid"]
        },
    })

    datasets = prepare_datasets(records)

    results = []
    for name, dataset in datasets.items():
        print(f"\n{'=' * 70}\nEVALUATING: {name}\n{'=' * 70}")
        results.append(evaluate(dataset, name))

        # Save after each evaluation so partial progress is retained.
        pd.DataFrame(results).to_csv(
            EVAL_DIR / "pooled_detection_results.csv",
            index=False,
        )
        save_json(EVAL_DIR / "pooled_detection_results.json", results)

    print("\nPOOLED DETECTION RESULTS")
    print(pd.DataFrame(results).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    if RUN_ROUTEWISE:
        route_rows = evaluate_routes(records, datasets)
        print("\nROUTE-WISE RESULTS")
        print(pd.DataFrame(route_rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        ))

    print("\nSaved evaluation:", EVAL_DIR)
    print(
        "No complete-pipeline latency or FPS was measured. "
        "Ultralytics speed outputs concern detector evaluation only."
    )


if __name__ == "__main__":
    main()