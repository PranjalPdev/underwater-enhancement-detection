import inspect
import json
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics

from tqdm import tqdm
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.models.yolo.detect.val import DetectionValidator


# =====================================================================
# FIXED INPUTS: DO NOT CHANGE EXPERIMENTAL SETTINGS HERE
# =====================================================================
RUN_DIR = Path(
    r"C:\internnn\temp_eval"
    r"\proposed_pipeline_20261019T203221_524771Z"
)

EVALUATION_DIR = RUN_DIR / (
    "detection_evaluation_20261020T042244_360547Z"
)

CONFIGURATIONS = [
    "raw",
    "fixed_five_layer",
    "fixed_residual",
    "dynamic_efficientnet",
    "resize_only",
]

CLASS_NAMES = [
    "holothurian",
    "echinus",
    "scallop",
    "starfish",
]

EXPECTED_IMAGES = 800

# Use the same installed Ultralytics version as the evaluation.
EXPECTED_ULTRALYTICS_VERSION = "8.4.98"

# 1,000 replicates for this analysis.
# This can take some time because pooled AP is recomputed each time.
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# Absolute AP tolerance, not percentage points.
# Do not enlarge merely to make a failed reconstruction pass.
REPRODUCTION_TOLERANCE = 0.0002

IOU_THRESHOLDS = torch.linspace(0.50, 0.95, 10)

ANALYSIS_ID = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%S_%fZ"
)
ANALYSIS_DIR = EVALUATION_DIR / (
    f"frozen_bootstrap_{ANALYSIS_ID}"
)

COMPARISONS = [
    ("dynamic_efficientnet", "raw"),
    ("dynamic_efficientnet", "fixed_residual"),
    ("dynamic_efficientnet", "fixed_five_layer"),
    ("resize_only", "raw"),
]


# =====================================================================
# FILE HELPERS
# =====================================================================
def save_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def prediction_path(configuration):
    return (
        EVALUATION_DIR
        / "ultralytics_results"
        / configuration
        / "predictions.json"
    )


def image_id_from_path(path):
    # Ultralytics JSON export converts numeric stems to integers.
    stem = Path(path).stem
    return str(int(stem)) if stem.isnumeric() else stem


def normalize_json_image_id(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    text = str(value)
    return str(int(text)) if text.isnumeric() else text


# =====================================================================
# READ THE FIXED MANIFEST
# =====================================================================
def load_manifest():
    path = RUN_DIR / "detection_manifest.json"
    records = json.loads(path.read_text(encoding="utf-8"))

    if len(records) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(records)}."
        )

    image_ids = []
    filenames = []
    dimensions = []

    for record in records:
        filename = Path(record["path"]).name
        image_id = image_id_from_path(filename)

        height = record.get("height")
        width = record.get("width")

        if height is None or width is None:
            frozen_image = (
                EVALUATION_DIR / "datasets" / "raw"
                / "images" / filename
            )
            image = cv2.imread(str(frozen_image))
            if image is None:
                raise RuntimeError(
                    f"Cannot read dimensions: {frozen_image}"
                )
            height, width = image.shape[:2]

        image_ids.append(image_id)
        filenames.append(filename)
        dimensions.append((int(height), int(width)))

    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError(
            "Image IDs are not unique after JSON-ID normalization."
        )

    return records, image_ids, filenames, dimensions


# =====================================================================
# GROUND-TRUTH LABELS
# =====================================================================
def load_labels(path, height, width):
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        values = np.asarray(line.split(), dtype=np.float32)

        if values.shape != (5,):
            raise ValueError(
                f"{path}:{number}: expected five fields."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"{path}:{number}: non-finite annotation."
            )

        class_id, x, y, w, h = values

        if class_id != int(class_id) or not 0 <= class_id < 4:
            raise ValueError(f"Invalid class in {path}:{number}")

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            raise ValueError(f"Invalid box in {path}:{number}")

        rows.append(values)

    if not rows:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty((0, 4), dtype=torch.float32),
        )

    labels = np.asarray(rows, dtype=np.float32)

    # Match Ultralytics' duplicate-label handling.
    _, unique_indices = np.unique(
        labels, axis=0, return_index=True
    )
    if len(unique_indices) != len(labels):
        labels = labels[unique_indices]

    classes = torch.from_numpy(labels[:, 0].copy())

    xywh = labels[:, 1:5]
    boxes = np.empty_like(xywh)

    boxes[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2) * width
    boxes[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2) * height
    boxes[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2) * width
    boxes[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2) * height

    return classes, torch.from_numpy(boxes)

def verified_json_category_mapping():
    import re

    source = inspect.getsource(DetectionValidator.init_metrics)
    compact_source = re.sub(r"\s+", "", source)

    expected_expression = (
        "list(range(1,len(model.names)+1))"
    )

    if expected_expression not in compact_source:
        print("\nINSTALLED init_metrics() SOURCE:\n")
        print(source)
        raise RuntimeError(
            "Could not verify the one-based JSON export mapping "
            "from the installed Ultralytics source. "
            "Send the printed source before changing category IDs."
        )

    observed_ids = set()

    for configuration in CONFIGURATIONS:
        path = prediction_path(configuration)
        items = json.loads(path.read_text(encoding="utf-8"))

        for item in items:
            value = item["category_id"]
            numeric = float(value)

            if not numeric.is_integer():
                raise RuntimeError(
                    f"Non-integer JSON category ID in {path}: {value}"
                )

            observed_ids.add(int(numeric))

    mapping = {
        exported_id: exported_id - 1
        for exported_id in range(1, len(CLASS_NAMES) + 1)
    }

    unsupported = observed_ids - set(mapping)
    if unsupported:
        raise RuntimeError(
            f"Unexpected JSON category IDs: {sorted(unsupported)}. "
            f"Observed IDs: {sorted(observed_ids)}"
        )

    print("\nVerified JSON-to-detector category mapping:")
    for exported_id, detector_id in mapping.items():
        print(
            f"  JSON {exported_id} -> class {detector_id} "
            f"({CLASS_NAMES[detector_id]})"
        )

    save_json(
        ANALYSIS_DIR / "json_category_mapping.json",
        {
            "observed_export_ids": sorted(observed_ids),
            "export_id_to_detector_id": mapping,
            "verification": (
                "Installed DetectionValidator.init_metrics source "
                "uses one-based non-COCO category export."
            ),
        },
    )

    return mapping
# =====================================================================
# LOAD SAVED PREDICTIONS
# =====================================================================
def load_predictions(
    configuration,
    allowed_ids,
    category_mapping,):
    path = prediction_path(configuration)
    predictions = json.loads(path.read_text(encoding="utf-8"))

    grouped = {image_id: [] for image_id in allowed_ids}

    for item in predictions:
        image_id = normalize_json_image_id(item["image_id"])

        if image_id not in grouped:
            raise RuntimeError(
                f"Unknown image ID in {path}: {image_id}"
            )

        exported_category = int(item["category_id"])

        if exported_category not in category_mapping:
            raise RuntimeError(
        f"Unsupported exported category ID "
        f"{exported_category} in {path}")

        category = category_mapping[exported_category]

        x, y, width, height = map(float, item["bbox"])
        score = float(item["score"])

        values = [x, y, width, height, score]
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Non-finite prediction in {path}."
            )
        if width < 0 or height < 0 or not 0 <= score <= 1:
            raise RuntimeError(
                f"Invalid prediction in {path}: {item}"
            )

        grouped[image_id].append([
            x,
            y,
            x + width,
            y + height,
            score,
            category,
        ])

    arrays = {}
    for image_id, rows in grouped.items():
        arrays[image_id] = (
            np.asarray(rows, dtype=np.float32)
            if rows
            else np.empty((0, 6), dtype=np.float32)
        )

    return arrays


# =====================================================================
# PER-IMAGE MATCHING, USING INSTALLED ULTRALYTICS
# =====================================================================
def build_statistics(
    configuration,
    image_ids,
    ground_truth,
    category_mapping,
):
    predictions = load_predictions(
    configuration,
    set(image_ids),
    category_mapping,)
    # match_predictions uses self.iouv; no detector is initialized.
    matcher = object.__new__(BaseValidator)
    matcher.iouv = IOU_THRESHOLDS

    stats = []

    for image_id in image_ids:
        pred = predictions[image_id]
        true_classes, true_boxes = ground_truth[image_id]

        pred_boxes = torch.from_numpy(pred[:, :4].copy())
        pred_classes = torch.from_numpy(pred[:, 5].copy())

        if len(pred) and len(true_classes):
            ious = box_iou(true_boxes, pred_boxes)

            correct = matcher.match_predictions(
                pred_classes,
                true_classes,
                ious,
            ).cpu().numpy()
        else:
            correct = np.zeros(
                (len(pred), len(IOU_THRESHOLDS)),
                dtype=bool,
            )

        stats.append({
            "tp": correct,
            "conf": pred[:, 4].copy(),
            "pred_cls": pred[:, 5].copy(),
            "target_cls": true_classes.numpy().copy(),
        })

    return stats


def pooled_ap(statistics, indices):
    # Repeated indices intentionally repeat both detections and GT.
    # Matching is per image, so precomputed matches can be reused.
    selected = [statistics[int(i)] for i in indices]

    tp = np.concatenate([s["tp"] for s in selected], axis=0)
    conf = np.concatenate([s["conf"] for s in selected])
    pred_cls = np.concatenate([s["pred_cls"] for s in selected])
    target_cls = np.concatenate([s["target_cls"] for s in selected])

    if set(np.unique(target_cls).astype(int)) != set(range(4)):
        raise RuntimeError(
            "A resample omitted a ground-truth class. "
            "Do not silently change the classes averaged by mAP."
        )

    if not len(conf):
        return np.array([0.0, 0.0])

    result = ap_per_class(
        tp=tp,
        conf=conf,
        pred_cls=pred_cls,
        target_cls=target_cls,
        plot=False,
        names={i: name for i, name in enumerate(CLASS_NAMES)},
    )

    # Installed Ultralytics returns:
    # tp, fp, p, r, f1, ap, unique_classes, ...
    ap = np.asarray(result[5])

    if ap.shape != (4, len(IOU_THRESHOLDS)):
        raise RuntimeError(
            f"Unexpected AP output shape: {ap.shape}. "
            "Check the installed Ultralytics API."
        )

    return np.array([
        float(ap[:, 0].mean()),
        float(ap.mean()),
    ])


# =====================================================================
# FREEZE INPUT RECORDS WITHOUT MODIFYING THE EVALUATION
# =====================================================================
def freeze_records(filenames):
    snapshot = ANALYSIS_DIR / "record_snapshot"
    snapshot.mkdir()

    candidates = [
        EVALUATION_DIR / "pooled_detection_results.csv",
        EVALUATION_DIR / "pooled_detection_results.json",
        EVALUATION_DIR / "evaluation_configuration.json",
        RUN_DIR / "detection_manifest.json",
        RUN_DIR / "routing_manifest.csv",
        RUN_DIR / "configuration.json",
        RUN_DIR / "dataset_counts.json",
        RUN_DIR / "excluded_images.json",
        RUN_DIR / "training_manifest.json",
        RUN_DIR / "validation_manifest.json",
        RUN_DIR / "validation_report.txt",
        RUN_DIR / "label_cleanup_zero_height_v1.json",
    ]

    hashes = []

    for path in candidates:
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
            hashes.append({
                "path": str(path),
                "sha256": sha256_file(path),
            })

    for configuration in CONFIGURATIONS:
        path = prediction_path(configuration)
        hashes.append({
            "path": str(path),
            "sha256": sha256_file(path),
        })

    # Verify that each configuration used identical label files.
    for filename in filenames:
        label_name = f"{Path(filename).stem}.txt"
        raw_label = (
            EVALUATION_DIR / "datasets" / "raw"
            / "labels" / label_name
        )
        reference_hash = sha256_file(raw_label)

        hashes.append({
            "path": str(raw_label),
            "sha256": reference_hash,
        })

        for configuration in CONFIGURATIONS:
            label = (
                EVALUATION_DIR / "datasets" / configuration
                / "labels" / label_name
            )
            if sha256_file(label) != reference_hash:
                raise RuntimeError(
                    f"Label mismatch for {configuration}: {label_name}"
                )

    save_json(ANALYSIS_DIR / "input_hashes.json", hashes)

    # Preserve exact metric/matching code used by this installation.
    matching_source = inspect.getsource(
        BaseValidator.match_predictions
    )
    ap_source = inspect.getsource(ap_per_class)

    (snapshot / "installed_metric_functions.txt").write_text(
        matching_source + "\n\n" + ap_source,
        encoding="utf-8",
    )

    script_path = globals().get("__file__")
    if script_path and Path(script_path).is_file():
        shutil.copy2(
            script_path,
            snapshot / "bootstrap_script.py",
        )


# =====================================================================
# MAIN
# =====================================================================
def main():
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected Ultralytics {EXPECTED_ULTRALYTICS_VERSION}, "
            f"found {ultralytics.__version__}. "
            "Use the original evaluation environment."
        )

    results_path = (
        EVALUATION_DIR / "pooled_detection_results.csv"
    )
    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    for configuration in CONFIGURATIONS:
        path = prediction_path(configuration)
        if not path.is_file():
            raise FileNotFoundError(path)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=False)

    _, image_ids, filenames, dimensions = load_manifest()
    freeze_records(filenames)

    save_json(ANALYSIS_DIR / "analysis_configuration.json", {
        "evaluation_directory": str(EVALUATION_DIR),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "image_count": len(image_ids),
        "bootstrap_replicates": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "sampling_unit": "image",
        "paired": True,
        "interval": "95% percentile",
        "scope": (
            "Conditional on the saved predictions and evaluated images; "
            "not training-seed or dataset-provenance uncertainty."
        ),
    })

    print("[1/4] Reading frozen evaluation annotations...")

    ground_truth = {}
    for image_id, filename, (height, width) in zip(
        image_ids, filenames, dimensions
    ):
        label_path = (
            EVALUATION_DIR / "datasets" / "raw"
            / "labels" / f"{Path(filename).stem}.txt"
        )
        ground_truth[image_id] = load_labels(
            label_path, height, width
        )

    print("[2/4] Matching saved detections to annotations...")
    category_mapping = verified_json_category_mapping()
    
    statistics = {
    configuration: build_statistics(
        configuration,
        image_ids,
        ground_truth,
        category_mapping,
    )
    for configuration in CONFIGURATIONS
}

    original_results = pd.read_csv(results_path).set_index(
        "configuration"
    )
    full_indices = np.arange(len(image_ids))

    point_estimates = {}
    reproduction_rows = []

    for configuration in CONFIGURATIONS:
        point = pooled_ap(
            statistics[configuration], full_indices
        )
        point_estimates[configuration] = point

        recorded = original_results.loc[
            configuration, ["map50", "map50_95"]
        ].to_numpy(dtype=float)

        differences = point - recorded

        reproduction_rows.append({
            "configuration": configuration,
            "recorded_map50": recorded[0],
            "reconstructed_map50": point[0],
            "difference_map50": differences[0],
            "recorded_map50_95": recorded[1],
            "reconstructed_map50_95": point[1],
            "difference_map50_95": differences[1],
            "passes": bool(
                np.all(
                    np.abs(differences)
                    <= REPRODUCTION_TOLERANCE
                )
            ),
        })

    reproduction = pd.DataFrame(reproduction_rows)
    reproduction.to_csv(
        ANALYSIS_DIR / "metric_reproduction.csv",
        index=False,
    )

    print("\nOFFLINE METRIC REPRODUCTION")
    print(reproduction.to_string(
        index=False,
        float_format=lambda x: f"{x:.7f}",
    ))

    if not reproduction["passes"].all():
        raise RuntimeError(
            "\nOffline AP did not reproduce the recorded evaluation "
            "within tolerance. Bootstrap was NOT performed.\n"
            "Send metric_reproduction.csv for diagnosis. "
            "Do not enlarge the tolerance or alter predictions "
            "just to make the check pass."
        )

    print("\n[3/4] Running paired image bootstrap...")

    rng = np.random.default_rng(RANDOM_SEED)

    samples = {
        configuration: np.empty((N_BOOTSTRAP, 2))
        for configuration in CONFIGURATIONS
    }

    for replicate in tqdm(
        range(N_BOOTSTRAP), desc="Paired bootstrap"
    ):
        indices = rng.integers(
            0, len(image_ids), size=len(image_ids)
        )

        for configuration in CONFIGURATIONS:
            samples[configuration][replicate] = pooled_ap(
                statistics[configuration], indices
            )

    np.savez_compressed(
        ANALYSIS_DIR / "bootstrap_samples.npz",
        **samples,
    )

    print("[4/4] Saving confidence intervals...")

    metric_names = ["map50", "map50_95"]
    configuration_rows = []

    for configuration in CONFIGURATIONS:
        for metric_index, metric_name in enumerate(metric_names):
            lower, upper = np.percentile(
                samples[configuration][:, metric_index],
                [2.5, 97.5],
            )

            configuration_rows.append({
                "configuration": configuration,
                "metric": metric_name,
                "point_estimate": (
                    point_estimates[configuration][metric_index]
                ),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "replicates": N_BOOTSTRAP,
            })

    configuration_table = pd.DataFrame(configuration_rows)
    configuration_table.to_csv(
        ANALYSIS_DIR / "configuration_intervals.csv",
        index=False,
    )

    comparison_rows = []

    for first, second in COMPARISONS:
        for metric_index, metric_name in enumerate(metric_names):
            paired_differences = (
                samples[first][:, metric_index]
                - samples[second][:, metric_index]
            )

            lower, upper = np.percentile(
                paired_differences, [2.5, 97.5]
            )

            comparison_rows.append({
                "comparison": f"{first} minus {second}",
                "metric": metric_name,
                "observed_difference": (
                    point_estimates[first][metric_index]
                    - point_estimates[second][metric_index]
                ),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "interval_contains_zero": bool(
                    lower <= 0 <= upper
                ),
                "replicates": N_BOOTSTRAP,
            })

    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(
        ANALYSIS_DIR / "paired_difference_intervals.csv",
        index=False,
    )

    print("\nCONFIGURATION INTERVALS")
    print(configuration_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))

    print("\nPAIRED DIFFERENCE INTERVALS")
    print(comparison_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))

    print("\nSaved analysis:", ANALYSIS_DIR)
    print(
        "\nOriginal evaluation files were not modified. "
        "Intervals describe image-resampling uncertainty only. "
        "They do not resolve near-duplicate dependence, "
        "checkpoint provenance, or possible test-informed tuning."
    )


if __name__ == "__main__":
    main()