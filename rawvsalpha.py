
from pathlib import Path
import csv
import hashlib
import shutil
import math
import yaml

from ultralytics import YOLO


# ============================================================
# 1. PATHS: CHECK THESE BEFORE RUNNING
# ============================================================

RAW_IMAGES = Path(r"C:\internnn\URPC2020\images\test")

ALPHA0_IMAGES = Path(
    r"C:\internnn\alpha_sweep\alpha_0_00\images"
)

# This is the cleaned label directory used by the frozen benchmark.
CLEAN_LABELS = Path(
    r"C:\internnn\temp_eval\proposed_pipeline_20261019T203221_524771Z"
    r"\detection_evaluation_20261020T042244_360547Z"
    r"\datasets\raw\labels"
)
DETECTOR_WEIGHTS = Path(
    r"C:\Users\pppan\runs\detect\urpc2020_yolov8n-9\weights\best.pt"
)

# Choose a new output directory that does not already exist.
RUN_ROOT = Path(
    r"C:\internnn\controlled_raw_alpha0_audit_20261025"
)

EXPECTED_IMAGES = 800
EXPECTED_BOXES = 6579

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

CLASS_NAMES = [
    "holothurian",  # class 0
    "echinus",      # class 1
    "scallop",      # class 2
    "starfish",     # class 3
]


# ============================================================
# 2. SAFETY CHECKS
# ============================================================

for folder in (RAW_IMAGES, ALPHA0_IMAGES, CLEAN_LABELS):
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

if not DETECTOR_WEIGHTS.is_file():
    raise FileNotFoundError(
        "Set DETECTOR_WEIGHTS to the actual frozen YOLO .pt checkpoint.\n"
        f"Current value: {DETECTOR_WEIGHTS}"
    )

if RUN_ROOT.exists():
    raise FileExistsError(
        f"Output directory already exists:\n{RUN_ROOT}\n"
        "Change RUN_ROOT to a new directory so existing results are preserved."
    )


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_images(folder: Path) -> dict:
    """Index images recursively by case-insensitive file stem."""
    indexed = {}

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = path.stem.lower()
        if key in indexed:
            raise ValueError(
                f"Duplicate image stem '{path.stem}' in {folder}:\n"
                f"{indexed[key]}\n{path}"
            )
        indexed[key] = path

    return indexed


def index_labels(folder: Path) -> dict:
    """Index label text files by case-insensitive file stem."""
    indexed = {}

    for path in folder.glob("*.txt"):
        key = path.stem.lower()
        if key in indexed:
            raise ValueError(f"Duplicate label stem: {path.stem}")
        indexed[key] = path

    return indexed


def validate_clean_labels(label_paths: dict) -> int:
    """Validate the common label set without modifying it."""
    total_boxes = 0

    for label_path in label_paths.values():
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Malformed label at {label_path}:{line_number}: {line}"
                )

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric label at {label_path}:{line_number}: {line}"
                ) from exc

            if class_id not in range(len(CLASS_NAMES)):
                raise ValueError(
                    f"Unexpected class ID at {label_path}:{line_number}: "
                    f"{class_id}"
                )

            if not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError(
                    f"Non-finite coordinate at {label_path}:{line_number}"
                )

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Non-positive box at {label_path}:{line_number}: {line}"
                )

            total_boxes += 1

    return total_boxes


def write_dataset_yaml(dataset_root: Path) -> Path:
    config = {
        "path": dataset_root.as_posix(),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = dataset_root / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    return yaml_path


# ============================================================
# 4. INDEX AND CHECK THE 800 IMAGES
# ============================================================

print("\n[1/6] Indexing images and labels...")

raw_index = index_images(RAW_IMAGES)
alpha_index = index_images(ALPHA0_IMAGES)
label_index = index_labels(CLEAN_LABELS)

print(f"Raw images:       {len(raw_index)}")
print(f"Alpha-zero images:{len(alpha_index)}")
print(f"Clean label files:{len(label_index)}")

if len(raw_index) != EXPECTED_IMAGES:
    raise ValueError(f"Expected {EXPECTED_IMAGES} raw images.")

if len(alpha_index) != EXPECTED_IMAGES:
    raise ValueError(f"Expected {EXPECTED_IMAGES} alpha-zero images.")

if len(label_index) != EXPECTED_IMAGES:
    raise ValueError(f"Expected {EXPECTED_IMAGES} clean label files.")

if set(raw_index) != set(alpha_index):
    raise ValueError("Raw and alpha-zero image filename stems differ.")

if set(raw_index) != set(label_index):
    raise ValueError("Image filename stems do not match the clean labels.")

print("\nChecking image hashes...")

image_hash_mismatches = []

for stem in sorted(raw_index):
    raw_path = raw_index[stem]
    alpha_path = alpha_index[stem]

    if raw_path.suffix.lower() != alpha_path.suffix.lower():
        image_hash_mismatches.append((stem, "extension mismatch"))
        continue

    if sha256_file(raw_path) != sha256_file(alpha_path):
        image_hash_mismatches.append((stem, "SHA-256 mismatch"))

if image_hash_mismatches:
    print("Image mismatches found:")
    for item in image_hash_mismatches[:20]:
        print(item)
    raise RuntimeError(
        "Raw and alpha-zero images are not all byte-identical. "
        "Do not proceed with the identical-input test."
    )

print("PASS: all 800 raw and alpha-zero images are byte-identical.")


# ============================================================
# 5. VALIDATE LABELS AND BUILD ISOLATED DATASETS
# ============================================================

print("\n[2/6] Validating cleaned annotations...")

box_count = validate_clean_labels(label_index)
print(f"Valid annotation count: {box_count}")

if box_count != EXPECTED_BOXES:
    raise ValueError(
        f"Expected {EXPECTED_BOXES} valid boxes, got {box_count}."
    )

print("\n[3/6] Creating temporary datasets...")

raw_root = RUN_ROOT / "datasets" / "raw"
alpha_root = RUN_ROOT / "datasets" / "alpha0"

for dataset_root in (raw_root, alpha_root):
    (dataset_root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "test").mkdir(parents=True, exist_ok=True)

# Copy raw images and alpha-zero images using the same filenames.
# Copy the exact same cleaned label files into both datasets.
for stem in sorted(raw_index):
    raw_source = raw_index[stem]
    alpha_source = alpha_index[stem]
    label_source = label_index[stem]

    target_name = raw_source.name

    shutil.copy2(
        raw_source, raw_root / "images" / "test" / target_name
    )
    shutil.copy2(
        alpha_source, alpha_root / "images" / "test" / target_name
    )

    shutil.copy2(
        label_source, raw_root / "labels" / "test" / label_source.name
    )
    shutil.copy2(
        label_source, alpha_root / "labels" / "test" / label_source.name
    )

raw_yaml = write_dataset_yaml(raw_root)
alpha_yaml = write_dataset_yaml(alpha_root)

print(f"Raw dataset YAML:   {raw_yaml}")
print(f"Alpha-zero YAML:    {alpha_yaml}")
print("PASS: both temporary datasets use the same cleaned labels.")


# ============================================================
# 6. EVALUATE BOTH USING IDENTICAL SETTINGS
# ============================================================

print("\n[4/6] Checking checkpoint hash...")

checkpoint_hash_before = sha256_file(DETECTOR_WEIGHTS)
print(f"Checkpoint SHA-256: {checkpoint_hash_before}")

RESULTS_ROOT = RUN_ROOT / "results"
RESULTS_ROOT.mkdir(parents=True)

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

def evaluate(condition: str, yaml_path: Path) -> dict:
    print(f"\nEvaluating: {condition}")
    print(f"Dataset: {yaml_path}")

    # Reload the same checkpoint for each condition.
    model = YOLO(str(DETECTOR_WEIGHTS))

    metrics = model.val(
        data=str(yaml_path),
        project=str(RESULTS_ROOT),
        name=condition,
        exist_ok=True,
        **COMMON_ARGS,
    )

    box = metrics.box

    result = {
        "condition": condition,
        "images": EXPECTED_IMAGES,
        "annotations": EXPECTED_BOXES,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }

    print(
        f"{condition}: "
        f"P={result['precision']:.8f}, "
        f"R={result['recall']:.8f}, "
        f"mAP50={result['mAP50']:.8f}, "
        f"mAP50-95={result['mAP50_95']:.8f}"
    )

    return result


print("\n[5/6] Running raw evaluation...")
raw_result = evaluate("raw", raw_yaml)

print("\nRunning alpha-zero evaluation...")
alpha_result = evaluate("alpha0", alpha_yaml)

checkpoint_hash_after = sha256_file(DETECTOR_WEIGHTS)

if checkpoint_hash_before != checkpoint_hash_after:
    raise RuntimeError("Checkpoint file hash changed during evaluation.")


# ============================================================
# 7. SAVE AND COMPARE RESULTS
# ============================================================

print("\n[6/6] Comparing metrics...")

metric_names = ["precision", "recall", "mAP50", "mAP50_95"]

comparison = {
    "raw": raw_result,
    "alpha0": alpha_result,
    "differences_alpha0_minus_raw": {
        key: alpha_result[key] - raw_result[key]
        for key in metric_names
    },
}

csv_path = RUN_ROOT / "comparison.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition", "images", "annotations",
            "precision", "recall", "mAP50", "mAP50_95"
        ],
    )
    writer.writeheader()
    writer.writerow(raw_result)
    writer.writerow(alpha_result)

json_path = RUN_ROOT / "comparison.json"
import json
json_path.write_text(
    json.dumps(comparison, indent=2),
    encoding="utf-8",
)

print("\n========== FINAL COMPARISON ==========")

for key in metric_names:
    raw_value = raw_result[key]
    alpha_value = alpha_result[key]
    difference = alpha_value - raw_value

    print(
        f"{key:10s} | raw={raw_value:.10f} | "
        f"alpha0={alpha_value:.10f} | "
        f"difference={difference:+.12f}"
    )

same_metrics = all(
    abs(alpha_result[key] - raw_result[key]) <= 1e-12
    for key in metric_names
)

print("\nAll four metrics identical:", same_metrics)
print(f"CSV results:  {csv_path}")
print(f"JSON results: {json_path}")
print(f"Full outputs: {RUN_ROOT}")
