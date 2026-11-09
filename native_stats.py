import hashlib
import inspect
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ultralytics

from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class


# =====================================================================
# CONFIGURATION
# =====================================================================
RUN_DIR = Path(
    r"C:\internnn\temp_eval"
    r"\proposed_pipeline_20261019T203221_524771Z"
)

EVAL_DIR = RUN_DIR / (
    "detection_evaluation_20261020T042244_360547Z"
)

YOLO_WEIGHTS = Path(
    r"C:\Users\pppan\runs\detect"
    r"\urpc2020_yolov8n-9\weights\best.pt"
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

COMPARISONS = [
    ("dynamic_efficientnet", "raw"),
    ("dynamic_efficientnet", "fixed_residual"),
    ("dynamic_efficientnet", "fixed_five_layer"),
    ("resize_only", "raw"),
]

EXPECTED_VERSION = "8.4.98"
EXPECTED_IMAGES = 800
EXPECTED_INSTANCES = 6579

REPRODUCTION_TOLERANCE = 0.0002

N_BOOTSTRAP = 1000
RANDOM_SEED = 42

RUN_ID = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%S_%fZ"
)

ANALYSIS_DIR = EVAL_DIR / (
    f"native_statistics_bootstrap_{RUN_ID}"
)

LAST_CAPTURE_VALIDATOR = None
# =====================================================================
# UTILITIES
# =====================================================================
def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2),
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


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def check_class_mapping(model):
    actual = [
        model.names[index]
        for index in range(len(model.names))
    ]
    if actual != CLASS_NAMES:
        raise RuntimeError(
            f"Class mapping mismatch:\n"
            f"Expected: {CLASS_NAMES}\n"
            f"Actual:   {actual}"
        )


# =====================================================================
# CAPTURE EXACT PER-IMAGE VALIDATOR STATISTICS
# =====================================================================
class CaptureDetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        global LAST_CAPTURE_VALIDATOR

        super().__init__(*args, **kwargs)
        self.native_records = []

        # Retain the instance created internally by model.val().
        LAST_CAPTURE_VALIDATOR = self

    def _process_batch(self, preds, batch):
        result = super()._process_batch(preds, batch)

        self.native_records.append({
            "filename": Path(batch["im_file"]).name,
            "tp": as_numpy(result["tp"]).astype(bool),
            "conf": as_numpy(preds["conf"]).reshape(-1),
            "pred_cls": as_numpy(preds["cls"]).reshape(-1),
            "target_cls": as_numpy(batch["cls"]).reshape(-1),
        })

        return result

# =====================================================================
# POOLED AP FROM NATIVE STATISTICS
# =====================================================================
def pooled_ap(records, indices):
    selected = [records[int(index)] for index in indices]

    tp = np.concatenate(
        [record["tp"] for record in selected], axis=0
    )
    conf = np.concatenate(
        [record["conf"] for record in selected]
    )
    pred_cls = np.concatenate(
        [record["pred_cls"] for record in selected]
    )
    target_cls = np.concatenate(
        [record["target_cls"] for record in selected]
    )

    observed_classes = set(
        np.unique(target_cls).astype(int).tolist()
    )
    if observed_classes != set(range(4)):
        raise RuntimeError(
            "A resample omitted a ground-truth class. "
            "Stopping rather than changing the mAP class set."
        )

    if not len(conf):
        return np.array([0.0, 0.0])

    result = ap_per_class(
        tp=tp,
        conf=conf,
        pred_cls=pred_cls,
        target_cls=target_cls,
        plot=False,
        names={
            index: name
            for index, name in enumerate(CLASS_NAMES)
        },
    )

    ap = np.asarray(result[5])

    if ap.shape != (4, 10):
        raise RuntimeError(
            f"Unexpected AP shape: {ap.shape}"
        )

    return np.array([
        float(ap[:, 0].mean()),
        float(ap.mean()),
    ])


# =====================================================================
# VERIFY FROZEN INPUTS
# =====================================================================
def verify_inputs():
    required = [
        YOLO_WEIGHTS,
        EVAL_DIR / "pooled_detection_results.csv",
        EVAL_DIR / "evaluation_configuration.json",
        RUN_DIR / "detection_manifest.json",
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(
        (EVAL_DIR / "evaluation_configuration.json")
        .read_text(encoding="utf-8")
    )

    if (
        sha256_file(YOLO_WEIGHTS)
        != config["yolo_checkpoint_sha256"]
    ):
        raise RuntimeError(
            "Detector checkpoint differs from the frozen evaluation."
        )

    expected_settings = {
        "imgsz": 640,
        "rect": False,
        "batch": 16,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
    }

    for key, expected in expected_settings.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"Frozen setting differs: {key}="
                f"{config.get(key)!r}, expected {expected!r}"
            )

    manifest = json.loads(
        (RUN_DIR / "detection_manifest.json")
        .read_text(encoding="utf-8")
    )
    filenames = [
        Path(record["path"]).name
        for record in manifest
    ]

    if len(filenames) != EXPECTED_IMAGES:
        raise RuntimeError("Unexpected manifest image count.")

    if len(set(filenames)) != len(filenames):
        raise RuntimeError("Duplicate manifest filenames.")

    audit_rows = []

    for filename in tqdm(
        filenames, desc="Checking frozen labels"
    ):
        label_name = f"{Path(filename).stem}.txt"
        reference = (
            EVAL_DIR / "datasets" / "raw"
            / "labels" / label_name
        )

        if not reference.is_file():
            raise FileNotFoundError(reference)

        reference_hash = sha256_file(reference)

        for configuration in CONFIGURATIONS:
            image = (
                EVAL_DIR / "datasets" / configuration
                / "images" / filename
            )
            label = (
                EVAL_DIR / "datasets" / configuration
                / "labels" / label_name
            )

            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(
                    f"Missing image or label for {configuration}: "
                    f"{filename}"
                )

            if sha256_file(label) != reference_hash:
                raise RuntimeError(
                    f"Label mismatch: {configuration}/{label_name}"
                )

        audit_rows.append({
            "filename": filename,
            "label_sha256": reference_hash,
        })

    save_json(
        ANALYSIS_DIR / "label_hash_audit.json",
        audit_rows,
    )

    return filenames

def save_native_records(path, records):
    if not records:
        raise RuntimeError(
            "Cannot save native statistics: records are empty."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pred_lengths = np.asarray(
        [len(record["conf"]) for record in records],
        dtype=np.int64,
    )

    target_lengths = np.asarray(
        [len(record["target_cls"]) for record in records],
        dtype=np.int64,
    )

    pred_offsets = np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(pred_lengths),
    ])

    target_offsets = np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(target_lengths),
    ])

    for record in records:
        n_predictions = len(record["conf"])

        if np.asarray(record["tp"]).shape != (n_predictions, 10):
            raise ValueError(
                f"Invalid TP array for {record['filename']}: "
                f"{np.asarray(record['tp']).shape}"
            )

        if len(record["pred_cls"]) != n_predictions:
            raise ValueError(
                f"Prediction count mismatch for "
                f"{record['filename']}"
            )

    np.savez_compressed(
        path,
        filenames=np.asarray(
            [record["filename"] for record in records],
            dtype=str,
        ),
        tp=np.concatenate(
            [
                np.asarray(record["tp"], dtype=bool)
                for record in records
            ],
            axis=0,
        ),
        conf=np.concatenate(
            [np.asarray(record["conf"]) for record in records]
        ),
        pred_cls=np.concatenate(
            [np.asarray(record["pred_cls"]) for record in records]
        ),
        target_cls=np.concatenate(
            [np.asarray(record["target_cls"]) for record in records]
        ),
        pred_offsets=pred_offsets,
        target_offsets=target_offsets,
    )

    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"Statistics file was not saved: {path}")
    print(
        f"[SAVED] Native statistics: {path}\n"
        f"        Images: {len(records)}, "
        f"predictions: {int(pred_offsets[-1])}, "
        f"targets: {int(target_offsets[-1])}"
    )

# =====================================================================
# REPLAY VALIDATION ON EXISTING GENERATED IMAGES
# =====================================================================
def capture_configuration(configuration):
    global LAST_CAPTURE_VALIDATOR
    LAST_CAPTURE_VALIDATOR = None



    # Keep the rest of your existing YAML creation,
    # model loading, and model.val(...) call unchanged.
    dataset = EVAL_DIR / "datasets" / configuration

    yaml_path = ANALYSIS_DIR / f"{configuration}.yaml"
    save_json(yaml_path, {
        "path": dataset.resolve().as_posix(),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": {
            index: name
            for index, name in enumerate(CLASS_NAMES)
        },
    })

    model = YOLO(str(YOLO_WEIGHTS))
    check_class_mapping(model)

    metrics = model.val(
        validator=CaptureDetectionValidator,
        data=str(yaml_path),
        split="test",
        imgsz=640,
        rect=False,
        batch=16,
        device="cpu",
        workers=0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        half=False,
        augment=False,
        plots=False,
        save_json=False,
        save_txt=False,
        project=str(ANALYSIS_DIR / "validation_replays"),
        name=configuration,
        exist_ok=False,
        verbose=True,
    )
    captured_validator = LAST_CAPTURE_VALIDATOR

    if captured_validator is None:
         raise RuntimeError(
            "The custom validator was not instantiated."
    )

    records = captured_validator.native_records

    if not records:
     raise RuntimeError(
        "The custom validator captured no per-image statistics."
    )
    if len(records) != EXPECTED_IMAGES:
        raise RuntimeError(
        f"{configuration}: captured {len(records)} images, "
        f"expected {EXPECTED_IMAGES}."
    )

    instance_count = sum(
        len(record["target_cls"]) for record in records)

    if instance_count != EXPECTED_INSTANCES:
        raise RuntimeError(
        f"{configuration}: {instance_count} instances, "
        f"expected {EXPECTED_INSTANCES}."
    )

    save_native_records(
    ANALYSIS_DIR / f"{configuration}_native_stats.npz",
    records,
)    
    return records, np.array([
        float(metrics.box.map50),
        float(metrics.box.map),
    ])


# =====================================================================
# MAIN
# =====================================================================
def main():
    if ultralytics.__version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"Use Ultralytics {EXPECTED_VERSION}; found "
            f"{ultralytics.__version__}."
        )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=False)

    filenames = verify_inputs()

    snapshot = ANALYSIS_DIR / "record_snapshot"
    snapshot.mkdir()

    for path in [
        EVAL_DIR / "pooled_detection_results.csv",
        EVAL_DIR / "pooled_detection_results.json",
        EVAL_DIR / "evaluation_configuration.json",
        RUN_DIR / "detection_manifest.json",
        RUN_DIR / "routing_manifest.csv",
    ]:
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)

    (snapshot / "native_matching_source.txt").write_text(
        inspect.getsource(DetectionValidator._process_batch)
        + "\n\n"
        + inspect.getsource(DetectionValidator.update_metrics)
        + "\n\n"
        + inspect.getsource(ap_per_class),
        encoding="utf-8",
    )

    script_path = globals().get("__file__")
    if script_path and Path(script_path).is_file():
        shutil.copy2(
            script_path,
            snapshot / "capture_and_bootstrap_script.py",
        )

    save_json(ANALYSIS_DIR / "analysis_configuration.json", {
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "sampling_unit": "image",
        "paired": True,
        "interval": "95% percentile",
        "original_evaluation": str(EVAL_DIR),
    })

    frozen = pd.read_csv(
        EVAL_DIR / "pooled_detection_results.csv"
    ).set_index("configuration")

    all_records = {}
    points = {}
    reproduction_rows = []

    for configuration in CONFIGURATIONS:
        print(
            f"\n{'=' * 72}\n"
            f"CAPTURING NATIVE STATISTICS: {configuration}\n"
            f"{'=' * 72}"
        )

        records, replay_point = capture_configuration(
            configuration
        )

        lookup = {
            record["filename"]: record
            for record in records
        }

        if len(lookup) != len(records):
            raise RuntimeError("Duplicate captured filenames.")

        if set(lookup) != set(filenames):
            raise RuntimeError(
                "Captured image identities differ from the manifest."
            )

        # Check AP using the validator's original image order.
        native_point = pooled_ap(
            records, np.arange(len(records))
        )

        # Canonical order aligns images across configurations.
        ordered_records = [
            lookup[filename] for filename in filenames
        ]
        ordered_point = pooled_ap(
            ordered_records,
            np.arange(len(ordered_records)),
        )

        recorded_point = frozen.loc[
            configuration, ["map50", "map50_95"]
        ].to_numpy(dtype=float)

        error = max(
            float(np.max(np.abs(replay_point - recorded_point))),
            float(np.max(np.abs(native_point - replay_point))),
            float(np.max(np.abs(ordered_point - replay_point))),
        )

        reproduction_rows.append({
            "configuration": configuration,
            "recorded_map50": recorded_point[0],
            "replay_map50": replay_point[0],
            "native_map50": native_point[0],
            "ordered_map50": ordered_point[0],
            "recorded_map50_95": recorded_point[1],
            "replay_map50_95": replay_point[1],
            "native_map50_95": native_point[1],
            "ordered_map50_95": ordered_point[1],
            "maximum_absolute_error": error,
            "passes": error <= REPRODUCTION_TOLERANCE,
        })

        pd.DataFrame(reproduction_rows).to_csv(
            ANALYSIS_DIR / "native_metric_reproduction.csv",
            index=False,
        )

        all_records[configuration] = ordered_records
        points[configuration] = ordered_point

    reproduction = pd.DataFrame(reproduction_rows)

    print("\nNATIVE METRIC REPRODUCTION")
    print(reproduction.to_string(
        index=False,
        float_format=lambda value: f"{value:.7f}",
    ))

    if not reproduction["passes"].all():
        raise RuntimeError(
            "Native-statistic reproduction failed. "
            "Bootstrap was not performed. "
            "Do not enlarge the tolerance; send the reproduction table."
        )

    # Ground-truth labels must agree per image, not only in total.
    for index, filename in enumerate(filenames):
        reference = all_records["raw"][index]["target_cls"]
        for configuration in CONFIGURATIONS:
            target = all_records[configuration][index]["target_cls"]
            if not np.array_equal(
                np.sort(reference), np.sort(target)
            ):
                raise RuntimeError(
                    f"Per-image target mismatch: {filename}"
                )

    print("\nRunning paired image bootstrap...")

    rng = np.random.default_rng(RANDOM_SEED)
    samples = {
        configuration: np.empty((N_BOOTSTRAP, 2))
        for configuration in CONFIGURATIONS
    }

    for replicate in tqdm(
        range(N_BOOTSTRAP), desc="Paired bootstrap"
    ):
        indices = rng.integers(
            0, len(filenames), size=len(filenames)
        )

        for configuration in CONFIGURATIONS:
            samples[configuration][replicate] = pooled_ap(
                all_records[configuration], indices
            )

    np.savez_compressed(
        ANALYSIS_DIR / "bootstrap_samples.npz",
        **samples,
    )

    metric_names = ["map50", "map50_95"]

    configuration_rows = []
    for configuration in CONFIGURATIONS:
        for metric_index, metric in enumerate(metric_names):
            lower, upper = np.percentile(
                samples[configuration][:, metric_index],
                [2.5, 97.5],
            )

            configuration_rows.append({
                "configuration": configuration,
                "metric": metric,
                "point_estimate": points[configuration][metric_index],
                "ci95_lower": lower,
                "ci95_upper": upper,
            })

    configuration_table = pd.DataFrame(configuration_rows)
    configuration_table.to_csv(
        ANALYSIS_DIR / "configuration_intervals.csv",
        index=False,
    )

    comparison_rows = []
    for first, second in COMPARISONS:
        for metric_index, metric in enumerate(metric_names):
            differences = (
                samples[first][:, metric_index]
                - samples[second][:, metric_index]
            )
            lower, upper = np.percentile(
                differences, [2.5, 97.5]
            )

            comparison_rows.append({
                "comparison": f"{first} minus {second}",
                "metric": metric,
                "observed_difference": (
                    points[first][metric_index]
                    - points[second][metric_index]
                ),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "interval_contains_zero": bool(
                    lower <= 0 <= upper
                ),
            })

    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(
        ANALYSIS_DIR / "paired_difference_intervals.csv",
        index=False,
    )

    print("\nCONFIGURATION INTERVALS")
    print(configuration_table.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))

    print("\nPAIRED DIFFERENCE INTERVALS")
    print(comparison_table.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))

    print("\nSaved analysis:", ANALYSIS_DIR)
    print(
        "\nOriginal labels, images, predictions, and result tables "
        "were not edited. Validation may refresh dataset cache files. "
        "Intervals describe image-resampling uncertainty only, "
        "not training variability or scene-level independence."
    )


if __name__ == "__main__":
    main()