import ast
import csv
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator


ROOT = Path(r"C:\internnn")

RUN_DIR = (
    ROOT
    / "temp_eval"
    / "proposed_pipeline_20261019T203221_524771Z"
)

BENCHMARK_DIR = (
    RUN_DIR / "detection_evaluation_20261020T042244_360547Z"
)

BENCHMARK_SOURCE = ROOT / "evaluate_efficientnet_pipeline.py"
MANIFEST = RUN_DIR / "detection_manifest.json"
CONFIGURATION = RUN_DIR / "configuration.json"

RAW_DATASET = BENCHMARK_DIR / "datasets" / "raw"
RESIDUAL_DATASET = BENCHMARK_DIR / "datasets" / "fixed_residual"

RESIDUAL_WEIGHTS = ROOT / "checkpoints" / "uwcnn_best.pth"
DETECTOR_WEIGHTS = Path(
    r"C:\Users\pppan\runs\detect"
    r"\urpc2020_yolov8n-9\weights\best.pt"
)

CLASS_NAMES = ["holothurian", "echinus", "scallop", "starfish"]
EXPECTED_IMAGES = 800
EXPECTED_INSTANCES = 6579
DEVICE = torch.device("cpu")

VAL_ARGS = {
    "split": "test",
    "imgsz": 640,
    "rect": False,
    "batch": 16,
    "device": "cpu",
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
    "augment": False,
    "plots": False,
    "save_json": False,
    "save_txt": False,
    "verbose": True,
}

REFERENCE_MAP50 = {
    "raw_alpha_0": 0.688624,
    "frozen_alpha_035": 0.644452,
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory(folder, extensions):
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    files = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        key = path.name.casefold()
        if key in files:
            raise ValueError(
                f"Duplicate filename:\n{files[key]}\n{path}"
            )
        files[key] = path

    return files


def read_labels(path):
    records = []

    for line_number, text in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not text.strip():
            continue

        values = np.asarray(text.split(), dtype=float)
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError(f"Invalid label: {path}:{line_number}")

        c, x, y, w, h = values

        if c != int(c) or not 0 <= c < 4:
            raise ValueError(f"Invalid class: {path}:{line_number}")

        if not (
            0 <= x <= 1 and 0 <= y <= 1
            and 0 < w <= 1 and 0 < h <= 1
        ):
            raise ValueError(
                f"Invalid normalized box: {path}:{line_number}\n"
                "Stopped without editing labels."
            )

        records.append(tuple(values.tolist()))

    return records


def extract_benchmark_functions():
    source = BENCHMARK_SOURCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(BENCHMARK_SOURCE))

    required = [
        "DenseResidualEnhancementCNN",
        "load_enhancer",
        "enhance",
    ]

    nodes = []
    for name in required:
        matches = [
            node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name == name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one definition of {name}."
            )
        nodes.append(matches[0])

    extracted = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(extracted)

    namespace = {
        "__name__": "verified_benchmark_functions",
        "torch": torch,
        "nn": nn,
        "np": np,
        "cv2": cv2,
        "DEVICE": DEVICE,
        "ALPHA": 1.0,
    }

    exec(
        compile(extracted, str(BENCHMARK_SOURCE), "exec"),
        namespace,
    )

    extracted_text = "\n\n".join(
        ast.get_source_segment(source, node) or ast.unparse(node)
        for node in nodes
    )

    return namespace, extracted_text


def validate_sources():
    for path in [
        BENCHMARK_SOURCE,
        MANIFEST,
        CONFIGURATION,
        RESIDUAL_WEIGHTS,
        DETECTOR_WEIGHTS,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(
        CONFIGURATION.read_text(encoding="utf-8-sig")
    )

    expected_hash = config.get("residual_checkpoint_sha256")
    if not expected_hash:
        raise ValueError(
            "Recorded residual checkpoint hash is missing."
        )

    if sha256(RESIDUAL_WEIGHTS) != expected_hash:
        raise RuntimeError(
            "Residual checkpoint differs from configuration.json."
        )

    if float(config["residual_alpha"]) != 0.35:
        raise RuntimeError(
            "Recorded benchmark residual alpha is not 0.35."
        )

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8-sig")
    )
    if not isinstance(manifest, list) or len(manifest) != EXPECTED_IMAGES:
        raise ValueError("Expected an 800-record detection manifest.")

    raw_images = inventory(RAW_DATASET / "images", IMAGE_EXTENSIONS)
    residual_images = inventory(
        RESIDUAL_DATASET / "images", IMAGE_EXTENSIONS
    )
    raw_labels = inventory(RAW_DATASET / "labels", {".txt"})
    residual_labels = inventory(
        RESIDUAL_DATASET / "labels", {".txt"}
    )

    entries = []
    seen = set()
    stems = set()
    target_count = 0

    for record in manifest:
        original = Path(record["path"])
        key = original.name.casefold()
        stem = original.stem.casefold()

        if key in seen or stem in stems:
            raise ValueError("Duplicate image filename or label stem.")
        seen.add(key)
        stems.add(stem)

        if not original.is_file():
            raise FileNotFoundError(original)
        if key not in raw_images or key not in residual_images:
            raise ValueError(f"Missing frozen image: {original.name}")

        original_hash = sha256(original)
        if original_hash != record["file_sha256"]:
            raise RuntimeError(
                f"Original image differs from manifest: {original}"
            )
        if sha256(raw_images[key]) != original_hash:
            raise RuntimeError(
                f"Frozen raw image differs from original: {original.name}"
            )

        label_key = f"{stem}.txt"
        if label_key not in raw_labels or label_key not in residual_labels:
            raise ValueError(f"Missing evaluation label: {label_key}")

        a = read_labels(raw_labels[label_key])
        b = read_labels(residual_labels[label_key])
        if a != b:
            raise RuntimeError(
                f"Frozen raw/residual label contents differ: {label_key}"
            )

        target_count += len(a)

        entries.append({
            "filename": original.name,
            "raw": raw_images[key],
            "residual": residual_images[key],
            "label": raw_labels[label_key],
            "raw_sha256": original_hash,
            "residual_sha256": sha256(residual_images[key]),
            "label_sha256": sha256(raw_labels[label_key]),
            "instances": len(a),
        })

    if set(raw_images) != seen or set(residual_images) != seen:
        raise ValueError(
            "Frozen image sets do not exactly match the manifest."
        )

    if target_count != EXPECTED_INSTANCES:
        raise RuntimeError(
            f"Expected {EXPECTED_INSTANCES} instances; "
            f"found {target_count}. No labels were modified."
        )

    return entries


def make_dataset(output, name):
    dataset = output / "datasets" / name
    (dataset / "images").mkdir(parents=True, exist_ok=False)
    (dataset / "labels").mkdir()
    return dataset


def prepare_datasets(output, entries, namespace):
    datasets = {
        name: make_dataset(output, name)
        for name in [
            "raw_alpha_0",
            "frozen_alpha_035",
            "generated_alpha_1",
        ]
    }

    enhancer = namespace["load_enhancer"](
        namespace["DenseResidualEnhancementCNN"](),
        RESIDUAL_WEIGHTS,
    )
    enhance = namespace["enhance"]

    generated_records = []

    for i, entry in enumerate(entries, start=1):
        name = entry["filename"]

        image = cv2.imread(str(entry["raw"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode: {entry['raw']}")

        residual_image = cv2.imread(
            str(entry["residual"]), cv2.IMREAD_COLOR
        )
        if residual_image is None or residual_image.shape != image.shape:
            raise RuntimeError(
                f"Invalid frozen residual dimensions: {name}"
            )

        shutil.copy2(
            entry["raw"], datasets["raw_alpha_0"] / "images" / name
        )
        shutil.copy2(
            entry["residual"],
            datasets["frozen_alpha_035"] / "images" / name,
        )

        generated = enhance(enhancer, image)
        destination = datasets["generated_alpha_1"] / "images" / name

        if not cv2.imwrite(str(destination), generated):
            raise IOError(f"Image write failed: {destination}")

        for dataset in datasets.values():
            destination_label = (
                dataset / "labels" / entry["label"].name
            )
            shutil.copy2(entry["label"], destination_label)
            if sha256(destination_label) != entry["label_sha256"]:
                raise RuntimeError("Copied label hash mismatch.")

        generated_records.append({
            "filename": name,
            "raw_sha256": entry["raw_sha256"],
            "frozen_alpha035_sha256": entry["residual_sha256"],
            "generated_alpha1_sha256": sha256(destination),
            "label_sha256": entry["label_sha256"],
            "instances": entry["instances"],
        })

        if i % 50 == 0:
            print(f"Generated alpha=1: {i}/{len(entries)}", flush=True)

    write_csv(output / "artifact_manifest.csv", generated_records)
    return datasets


def evaluate(dataset, name, output):
    yaml_path = output / f"{name}.yaml"
    save_json(yaml_path, {
        "path": dataset.resolve().as_posix(),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": dict(enumerate(CLASS_NAMES)),
    })

    model = YOLO(str(DETECTOR_WEIGHTS))
    actual_names = [
        model.names[i] for i in range(len(model.names))
    ]
    if actual_names != CLASS_NAMES:
        raise RuntimeError(
            f"Detector class mapping mismatch: {actual_names}"
        )

    requested = {
        **VAL_ARGS,
        "data": str(yaml_path),
        "project": str(output / "validation_results"),
        "name": name,
        "exist_ok": False,
    }
    save_json(output / f"{name}_requested_args.json", requested)

    captured = {}

    class RetainedDetectionValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["validator"] = self

    metrics = model.val(
        validator=RetainedDetectionValidator,
        **requested,
    )

    validator = captured.get("validator")
    if validator is None:
        raise RuntimeError("The validation instance was not captured.")

    resolved_args = vars(validator.args)
    save_json(
        output / f"{name}_resolved_args.json",
        resolved_args,
    )

    dataloader_files = getattr(
        getattr(validator.dataloader, "dataset", None),
        "im_files",
        None,
    )
    if dataloader_files is not None:
        (output / f"{name}_image_order.txt").write_text(
            "\n".join(map(str, dataloader_files)) + "\n",
            encoding="utf-8",
        )

    return {
        "configuration": name,
        "alpha": {
            "raw_alpha_0": 0.0,
            "frozen_alpha_035": 0.35,
            "generated_alpha_1": 1.0,
        }[name],
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "map50_difference_from_reference": (
            float(metrics.box.map50) - REFERENCE_MAP50[name]
            if name in REFERENCE_MAP50 else None
        ),
    }


def main():
    print("Validating manifests, checkpoints, and frozen labels...",
          flush=True)
    entries = validate_sources()

    print(
        f"Validation passed: {len(entries)} images, "
        f"{sum(e['instances'] for e in entries)} instances.",
        flush=True,
    )

    namespace, extracted_text = extract_benchmark_functions()

    output = ROOT / (
        "verified_alpha_sweep_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)

    (output / "extracted_benchmark_functions.txt").write_text(
        extracted_text, encoding="utf-8"
    )

    metadata = {
        "status": "preparing",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "generation_device": str(DEVICE),
        "alpha_generation": 1.0,
        "benchmark_source": str(BENCHMARK_SOURCE),
        "benchmark_source_sha256": sha256(BENCHMARK_SOURCE),
        "script_sha256": sha256(Path(__file__)),
        "residual_checkpoint": str(RESIDUAL_WEIGHTS),
        "residual_checkpoint_sha256": sha256(RESIDUAL_WEIGHTS),
        "detector_checkpoint": str(DETECTOR_WEIGHTS),
        "detector_checkpoint_sha256": sha256(DETECTOR_WEIGHTS),
        "input_manifest_sha256": sha256(MANIFEST),
        "class_names": CLASS_NAMES,
        "requested_common_validation_args": VAL_ARGS,
        "annotation_policy": (
            "Frozen benchmark labels copied unchanged; "
            "no clipping or cleaning in this experiment."
        ),
        "scope": (
            "Harmonized artifact generation and detector evaluation. "
            "Does not establish historical checkpoint training provenance "
            "or independent alpha tuning."
        ),
    }
    save_json(output / "metadata.json", metadata)

    datasets = prepare_datasets(output, entries, namespace)
    results = []

    for name, dataset in datasets.items():
        print(f"\nEvaluating {name}", flush=True)
        result = evaluate(dataset, name, output)
        results.append(result)

        save_json(output / "detector_metrics.json", results)
        write_csv(output / "detector_metrics.csv", results)

        print(json.dumps(result, indent=2), flush=True)

        if name in REFERENCE_MAP50:
            difference = abs(
                result["map50"] - REFERENCE_MAP50[name]
            )
            if difference > 2e-4:
                metadata["status"] = "reference_reproduction_failed"
                metadata["failed_configuration"] = name
                save_json(output / "metadata.json", metadata)
                raise RuntimeError(
                    f"{name} differs from the reported reference by "
                    f"{difference:.9f}. Results were saved; stopped "
                    "before proceeding with further evaluations."
                )

    metadata["status"] = "complete"
    save_json(output / "metadata.json", metadata)

    print("\nVERIFIED COEFFICIENT SWEEP")
    for result in results:
        print(
            f"alpha={result['alpha']:.2f}: "
            f"mAP50={result['map50']:.9f}, "
            f"mAP50:95={result['map50_95']:.9f}"
        )

    print(f"\nSaved experiment: {output}")


if __name__ == "__main__":
    main()