
from pathlib import Path
import ast
import csv
import hashlib
import importlib.util
import json
import shutil
import sys

import cv2
import numpy as np
import yaml
import torch
import ultralytics
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

BENCHMARK_SCRIPT = Path(
    r"C:\internnn\evaluate_efficientnet_pipeline.py"
)

RAW_IMAGES = Path(r"C:\internnn\URPC2020\images\test")

OLD_SWEEP_IMAGES = Path(
    r"C:\internnn\end_to_end_evaluation\3_uwcnn\images"
)

FROZEN_RESIDUAL_IMAGES = Path(
    r"C:\internnn\temp_eval\proposed_pipeline_20261019T203221_524771Z"
    r"\detection_evaluation_20261020T042244_360547Z"
    r"\datasets\fixed_residual\images"
)

CLEAN_LABELS = Path(
    r"C:\internnn\temp_eval\proposed_pipeline_20261019T203221_524771Z"
    r"\detection_evaluation_20261020T042244_360547Z"
    r"\datasets\raw\labels"
)

# A new output directory. Change its name if it already exists.
OUTPUT_ROOT = Path(
    r"C:\internnn\residual_reproduction_audit_20261025_01"
)

EXPECTED_IMAGES = 800
EXPECTED_BOXES = 6579
EXPECTED_ALPHA = 0.35

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

CLASS_NAMES = [
    "holothurian",
    "echinus",
    "scallop",
    "starfish",
]


# ============================================================
# 2. HELPERS
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def index_images(folder):
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")

    result = {}

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = path.stem.lower()

        if key in result:
            raise ValueError(
                f"Duplicate image stem '{key}' in {folder}:\n"
                f"{result[key]}\n{path}"
            )

        result[key] = path

    return result


def index_labels(folder):
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Label directory not found: {folder}")

    result = {}

    for path in folder.glob("*.txt"):
        key = path.stem.lower()

        if key in result:
            raise ValueError(f"Duplicate label stem: {key}")

        result[key] = path

    return result


def validate_clean_labels(label_index):
    total = 0

    for path in label_index.values():
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                fields = line.split()

                if len(fields) != 5:
                    raise ValueError(
                        f"Malformed label: {path}:{line_number}"
                    )

                try:
                    class_id = int(fields[0])
                    x, y, w, h = map(float, fields[1:])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric label: {path}:{line_number}"
                    ) from exc

                if class_id not in range(len(CLASS_NAMES)):
                    raise ValueError(
                        f"Unexpected class ID: {path}:{line_number}"
                    )

                if not np.isfinite([x, y, w, h]).all():
                    raise ValueError(
                        f"Non-finite coordinates: {path}:{line_number}"
                    )

                if w <= 0 or h <= 0:
                    raise ValueError(
                        f"Non-positive box: {path}:{line_number}"
                    )

                total += 1

    return total


def has_main_guard(tree):
    """Check that importing the benchmark won't call main() via its guard."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue

        test = node.test

        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            continue

        return True

    return False


def load_original_benchmark(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    if not has_main_guard(tree):
        raise RuntimeError(
            "The original benchmark does not have a recognizable "
            "'if __name__ == \"__main__\"' guard. "
            "For safety, inspect the end of the source script before "
            "importing it."
        )

    spec = importlib.util.spec_from_file_location(
        "original_residual_benchmark",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the original benchmark module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def write_yaml(dataset_root):
    config = {
        "path": dataset_root.as_posix(),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    path = dataset_root / "dataset.yaml"

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return path


# ============================================================
# 3. COMPARE IMAGE FOLDERS
# ============================================================

def compare_image_folders(reference_index, candidate_index, comparison_name):
    if set(reference_index) != set(candidate_index):
        missing = sorted(set(reference_index) - set(candidate_index))
        extra = sorted(set(candidate_index) - set(reference_index))

        raise ValueError(
            f"{comparison_name}: filename stems differ.\n"
            f"Missing: {missing[:10]}\n"
            f"Extra: {extra[:10]}"
        )

    details = []

    for key in sorted(reference_index):
        reference = reference_index[key]
        candidate = candidate_index[key]

        ref_image = cv2.imread(str(reference), cv2.IMREAD_COLOR)
        cand_image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)

        if ref_image is None or cand_image is None:
            raise RuntimeError(f"Could not decode image {key}")

        same_shape = ref_image.shape == cand_image.shape

        pixel_identical = (
            same_shape and np.array_equal(ref_image, cand_image)
        )

        if same_shape:
            diff = np.abs(
                ref_image.astype(np.float32)
                - cand_image.astype(np.float32)
            )
            mean_abs_diff = float(diff.mean())
            max_abs_diff = float(diff.max())
        else:
            mean_abs_diff = None
            max_abs_diff = None

        reference_hash = sha256_file(reference)
        candidate_hash = sha256_file(candidate)

        details.append({
            "comparison": comparison_name,
            "image": key,
            "reference_path": str(reference),
            "candidate_path": str(candidate),
            "byte_identical": reference_hash == candidate_hash,
            "pixel_identical": pixel_identical,
            "same_shape": same_shape,
            "mean_absolute_pixel_difference": mean_abs_diff,
            "max_absolute_pixel_difference": max_abs_diff,
        })

    summary = {
        "comparison": comparison_name,
        "number_of_images": len(details),
        "byte_identical": sum(r["byte_identical"] for r in details),
        "pixel_identical": sum(r["pixel_identical"] for r in details),
        "pixel_different": sum(not r["pixel_identical"] for r in details),
        "shape_mismatches": sum(not r["same_shape"] for r in details),
    }

    return details, summary


# ============================================================
# 4. START: VALIDATE PATHS AND LOAD SOURCE IMPLEMENTATION
# ============================================================

if OUTPUT_ROOT.exists():
    raise FileExistsError(
        f"Output directory already exists:\n{OUTPUT_ROOT}\n"
        "Choose a new OUTPUT_ROOT. No existing files were overwritten."
    )

for path in [
    BENCHMARK_SCRIPT,
    RAW_IMAGES,
    OLD_SWEEP_IMAGES,
    FROZEN_RESIDUAL_IMAGES,
    CLEAN_LABELS,
]:
    if not path.exists():
        raise FileNotFoundError(path)

print("[1/7] Loading original benchmark implementation...")

benchmark = load_original_benchmark(BENCHMARK_SCRIPT)

if float(benchmark.ALPHA) != EXPECTED_ALPHA:
    raise RuntimeError(
        f"Original script alpha is {benchmark.ALPHA}, "
        f"not {EXPECTED_ALPHA}"
    )

RESIDUAL_WEIGHTS = Path(benchmark.RESIDUAL_WEIGHTS)
DETECTOR_WEIGHTS = Path(benchmark.YOLO_WEIGHTS)

if not RESIDUAL_WEIGHTS.is_file():
    raise FileNotFoundError(RESIDUAL_WEIGHTS)

if not DETECTOR_WEIGHTS.is_file():
    raise FileNotFoundError(DETECTOR_WEIGHTS)

# Verify that the residual checkpoint matches the frozen benchmark
# configuration, rather than merely trusting its filename.
configuration_path = (
    Path(benchmark.RUN_DIR) / "configuration.json"
)

if not configuration_path.is_file():
    raise FileNotFoundError(
        "The original configuration file is missing:\n"
        f"{configuration_path}\n"
        "Do not claim exact checkpoint provenance without verifying it."
    )

configuration = json.loads(
    configuration_path.read_text(encoding="utf-8")
)

actual_residual_hash = sha256_file(RESIDUAL_WEIGHTS)
expected_residual_hash = configuration.get(
    "residual_checkpoint_sha256"
)

if not expected_residual_hash:
    raise RuntimeError(
        "Residual checkpoint SHA-256 is not recorded in the original "
        "configuration. Stop and inspect the provenance."
    )

if actual_residual_hash.lower() != expected_residual_hash.lower():
    raise RuntimeError(
        "Residual checkpoint hash differs from the original configuration.\n"
        f"Current: {actual_residual_hash}\n"
        f"Expected: {expected_residual_hash}"
    )

if float(configuration.get("residual_alpha", -1)) != EXPECTED_ALPHA:
    raise RuntimeError(
        "The original configuration does not record alpha=0.35."
    )

actual_detector_hash = sha256_file(DETECTOR_WEIGHTS)
source_script_hash = sha256_file(BENCHMARK_SCRIPT)

print(f"Residual checkpoint: {RESIDUAL_WEIGHTS}")
print(f"Residual SHA-256: {actual_residual_hash}")
print(f"Detector checkpoint: {DETECTOR_WEIGHTS}")
print(f"Detector SHA-256: {actual_detector_hash}")
print("PASS: residual checkpoint and alpha match the recorded configuration.")


# ============================================================
# 5. INDEX DATA AND CHECK LABELS
# ============================================================

print("\n[2/7] Indexing images and labels...")

raw_index = index_images(RAW_IMAGES)
old_index = index_images(OLD_SWEEP_IMAGES)
frozen_index = index_images(FROZEN_RESIDUAL_IMAGES)
label_index = index_labels(CLEAN_LABELS)

for name, index in [
    ("raw", raw_index),
    ("old sweep", old_index),
    ("frozen residual", frozen_index),
]:
    print(f"{name}: {len(index)} images")

    if len(index) != EXPECTED_IMAGES:
        raise ValueError(
            f"{name}: expected {EXPECTED_IMAGES} images, got {len(index)}"
        )

if set(raw_index) != set(old_index):
    raise ValueError("Raw versus old-sweep filenames do not match.")

if set(raw_index) != set(frozen_index):
    raise ValueError("Raw versus frozen-residual filenames do not match.")

if set(raw_index) != set(label_index):
    raise ValueError("Raw images and cleaned label filenames do not match.")

for key in raw_index:
    for name, index in [
        ("old sweep", old_index),
        ("frozen residual", frozen_index),
    ]:
        if raw_index[key].suffix.lower() != index[key].suffix.lower():
            raise ValueError(
                f"Extension mismatch for {key} in {name}."
            )

box_count = validate_clean_labels(label_index)

print(f"Clean label files: {len(label_index)}")
print(f"Valid boxes: {box_count}")

if box_count != EXPECTED_BOXES:
    raise ValueError(
        f"Expected {EXPECTED_BOXES} valid boxes; got {box_count}"
    )


# ============================================================
# 6. REGENERATE WITH THE ORIGINAL ENHANCEMENT FUNCTION
# ============================================================

print("\n[3/7] Regenerating alpha=0.35 residual images...")

OUTPUT_ROOT.mkdir(parents=True)

fresh_images_dir = OUTPUT_ROOT / "fresh_residual" / "images"
fresh_images_dir.mkdir(parents=True)

# Reuse the original architecture, checkpoint loader, preprocessing,
# forward pass, and output conversion from evaluate_efficientnet_pipeline.py.
residual_model = benchmark.load_enhancer(
    benchmark.DenseResidualEnhancementCNN(),
    RESIDUAL_WEIGHTS,
)

for index, key in enumerate(sorted(raw_index), start=1):
    source = raw_index[key]
    source_image = benchmark.read_image(source)

    # This is the original benchmark's enhance() function.
    enhanced_image = benchmark.enhance(residual_model, source_image)

    if enhanced_image is None:
        raise RuntimeError(f"Enhancer returned None for {source.name}")

    if enhanced_image.shape != source_image.shape:
        raise RuntimeError(
            f"Output shape changed for {source.name}: "
            f"{source_image.shape} -> {enhanced_image.shape}"
        )

    if enhanced_image.dtype != np.uint8:
        raise RuntimeError(
            f"Expected uint8 output for {source.name}, "
            f"got {enhanced_image.dtype}"
        )

    destination = fresh_images_dir / source.name
    benchmark.write_image(destination, enhanced_image)

    if index % 100 == 0 or index == EXPECTED_IMAGES:
        print(f"Generated {index}/{EXPECTED_IMAGES}")

del residual_model

print(f"Fresh images saved to: {fresh_images_dir}")


# ============================================================
# 7. AUDIT IMAGES, THEN RUN CONTROLLED DETECTOR EVALUATIONS
# ============================================================

print("\n[4/7] Comparing old and freshly generated pixels...")

fresh_index = index_images(fresh_images_dir)

if set(raw_index) != set(fresh_index):
    raise ValueError("Freshly generated image filenames do not match raw.")

details = []
summaries = []

for candidate_name, candidate_index in [
    ("old_sweep_vs_frozen", old_index),
    ("fresh_vs_frozen", fresh_index),
]:
    rows, summary = compare_image_folders(
        frozen_index,
        candidate_index,
        candidate_name,
    )
    details.extend(rows)
    summaries.append(summary)
    print(summary)

with (OUTPUT_ROOT / "image_audit_details.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
    writer.writeheader()
    writer.writerows(details)

(OUTPUT_ROOT / "image_audit_summary.json").write_text(
    json.dumps(summaries, indent=2),
    encoding="utf-8",
)

# Create standardized evaluation datasets. All four conditions receive
# exactly the same cleaned label files.
print("\n[5/7] Preparing evaluation datasets...")

condition_indexes = {
    "raw": raw_index,
    "old_sweep": old_index,
    "frozen_residual": frozen_index,
    "fresh_residual": fresh_index,
}

datasets_root = OUTPUT_ROOT / "datasets"
datasets_root.mkdir()

yaml_paths = {}

for condition, image_index in condition_indexes.items():
    root = datasets_root / condition
    image_dir = root / "images" / "test"
    label_dir = root / "labels" / "test"

    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for key in sorted(raw_index):
        # Use matching filenames across all evaluation conditions.
        source_image = image_index[key]
        target_filename = raw_index[key].name

        shutil.copy2(
            source_image,
            image_dir / target_filename,
        )
        shutil.copy2(
            label_index[key],
            label_dir / label_index[key].name,
        )

    yaml_paths[condition] = write_yaml(root)

print("PASS: all conditions use identical cleaned labels.")


# Evaluate all conditions with identical detector settings.
print("\n[6/7] Running detector evaluations...")

metrics_root = OUTPUT_ROOT / "detector_results"
metrics_root.mkdir()

COMMON_ARGS = {
    "imgsz": 640,
    "split": "test",
    "rect": False,
    "batch": 16,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
    "augment": False,
    "cache": False,
    "plots": False,
    "save_json": False,
    "verbose": False,
}

evaluation_results = []

for condition, data_yaml in yaml_paths.items():
    print(f"\nEvaluating: {condition}")

    model = YOLO(str(DETECTOR_WEIGHTS))

    metrics = model.val(
        data=str(data_yaml),
        project=str(metrics_root),
        name=condition,
        exist_ok=False,
        **COMMON_ARGS,
    )

    row = {
        "condition": condition,
        "images": EXPECTED_IMAGES,
        "annotations": EXPECTED_BOXES,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    evaluation_results.append(row)

    print(
        f"P={row['precision']:.8f} | "
        f"R={row['recall']:.8f} | "
        f"mAP50={row['mAP50']:.8f} | "
        f"mAP50-95={row['mAP50_95']:.8f}"
    )

# Ensure the detector weights were not changed during the experiment.
if sha256_file(DETECTOR_WEIGHTS) != actual_detector_hash:
    raise RuntimeError("Detector checkpoint hash changed during evaluation.")

results_csv = OUTPUT_ROOT / "detector_metrics.csv"

with results_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition", "images", "annotations",
            "precision", "recall", "mAP50", "mAP50_95"
        ],
    )
    writer.writeheader()
    writer.writerows(evaluation_results)

results_json = OUTPUT_ROOT / "detector_metrics.json"
results_json.write_text(
    json.dumps(evaluation_results, indent=2),
    encoding="utf-8",
)

# Save provenance to make the experiment reproducible.
metadata = {
    "benchmark_script": str(BENCHMARK_SCRIPT),
    "benchmark_script_sha256": source_script_hash,
    "residual_checkpoint": str(RESIDUAL_WEIGHTS),
    "residual_checkpoint_sha256": actual_residual_hash,
    "detector_checkpoint": str(DETECTOR_WEIGHTS),
    "detector_checkpoint_sha256": actual_detector_hash,
    "alpha": EXPECTED_ALPHA,
    "images": EXPECTED_IMAGES,
    "valid_annotations": EXPECTED_BOXES,
    "class_names": CLASS_NAMES,
    "torch_version": torch.__version__,
    "ultralytics_version": ultralytics.__version__,
    "evaluation_arguments": COMMON_ARGS,
    "image_audit": summaries,
}

(OUTPUT_ROOT / "experiment_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n[7/7] FINAL SUMMARY")

for summary in summaries:
    print(
        f"{summary['comparison']}: "
        f"pixel-identical {summary['pixel_identical']}/"
        f"{summary['number_of_images']}, "
        f"byte-identical {summary['byte_identical']}/"
        f"{summary['number_of_images']}"
    )

print("\nDetector metrics:")

for row in evaluation_results:
    print(
        f"{row['condition']:18s} "
        f"P={row['precision']:.6f} "
        f"R={row['recall']:.6f} "
        f"mAP50={row['mAP50']:.6f} "
        f"mAP50-95={row['mAP50_95']:.6f}"
    )

by_condition = {row["condition"]: row for row in evaluation_results}

for metric in ("precision", "recall", "mAP50", "mAP50_95"):
    print(
        f"Fresh - frozen, {metric}: "
        f"{by_condition['fresh_residual'][metric] - by_condition['frozen_residual'][metric]:+.10f}"
    )

print(f"\nOutput folder: {OUTPUT_ROOT}")
print(f"Image audit:   {OUTPUT_ROOT / 'image_audit_summary.json'}")
print(f"Image details: {OUTPUT_ROOT / 'image_audit_details.csv'}")
print(f"Metrics CSV:   {results_csv}")
print(f"Metrics JSON:  {results_json}")
print(f"Metadata:      {OUTPUT_ROOT / 'experiment_metadata.json'}")
