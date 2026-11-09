import ast
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(r"C:\internnn")

RUN_DIR = (
    ROOT
    / "temp_eval"
    / "proposed_pipeline_20261019T203221_524771Z"
)

BENCHMARK_DIR = (
    RUN_DIR
    / "detection_evaluation_20261020T042244_360547Z"
)

SOURCE_SCRIPT = ROOT / "evaluate_efficientnet_pipeline.py"
SOURCE_MANIFEST = RUN_DIR / "detection_manifest.json"
CONFIGURATION = RUN_DIR / "configuration.json"

CHECKPOINT = ROOT / "checkpoints" / "waternet_best.pth"

FROZEN_RAW_IMAGES = BENCHMARK_DIR / "datasets" / "raw" / "images"
FROZEN_FIVE_IMAGES = (
    BENCHMARK_DIR / "datasets" / "fixed_five_layer" / "images"
)

EXPECTED_IMAGES = 800
DEVICE = torch.device("cpu")

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


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    images = {}

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = path.name.casefold()
        if key in images:
            raise ValueError(
                "Ambiguous duplicate filename:\n"
                f"{images[key]}\n{path}"
            )

        images[key] = path

    return images


def extract_functions():
    source = SOURCE_SCRIPT.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(SOURCE_SCRIPT))

    required = [
        "FiveLayerEnhancementCNN",
        "load_enhancer",
        "enhance",
        "read_image",
        "write_image",
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
                f"Expected exactly one definition of {name}; "
                f"found {len(matches)}."
            )

        nodes.append(matches[0])

    extracted = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(extracted)

    namespace = {
        "__name__": "five_layer_audit_functions",
        "torch": torch,
        "nn": nn,
        "np": np,
        "cv2": cv2,
        "DEVICE": DEVICE,
    }

    # Execute only the selected definitions, not the experiment script.
    exec(
        compile(extracted, str(SOURCE_SCRIPT), "exec"),
        namespace,
    )

    extracted_text = "\n\n".join(
        ast.get_source_segment(source, node) or ast.unparse(node)
        for node in nodes
    )

    return namespace, extracted_text


def validate_inputs():
    for path in [
        SOURCE_SCRIPT,
        SOURCE_MANIFEST,
        CONFIGURATION,
        CHECKPOINT,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    configuration = json.loads(
        CONFIGURATION.read_text(encoding="utf-8-sig")
    )

    recorded_hash = configuration.get(
        "five_layer_checkpoint_sha256"
    )
    if not recorded_hash:
        raise ValueError(
            "configuration.json does not contain "
            "five_layer_checkpoint_sha256."
        )

    checkpoint_hash = sha256(CHECKPOINT)

    if checkpoint_hash != recorded_hash:
        raise RuntimeError(
            "Five-layer checkpoint differs from the hash recorded "
            "for the routing run. No regeneration performed."
        )

    records = json.loads(
        SOURCE_MANIFEST.read_text(encoding="utf-8-sig")
    )

    if not isinstance(records, list):
        raise ValueError("Detection manifest must contain a list.")

    if len(records) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} manifest records, "
            f"found {len(records)}."
        )

    raw_images = inventory(FROZEN_RAW_IMAGES)
    frozen_images = inventory(FROZEN_FIVE_IMAGES)

    names = set()
    stems = set()
    verified = []

    for record in records:
        original = Path(record["path"])
        key = original.name.casefold()
        stem = original.stem.casefold()

        if key in names or stem in stems:
            raise ValueError(
                f"Duplicate image filename or label stem: {original.name}"
            )

        names.add(key)
        stems.add(stem)

        if not original.is_file():
            raise FileNotFoundError(original)

        if key not in raw_images:
            raise FileNotFoundError(
                f"Missing frozen raw image: {original.name}"
            )

        if key not in frozen_images:
            raise FileNotFoundError(
                f"Missing frozen five-layer image: {original.name}"
            )

        original_hash = sha256(original)

        if original_hash != record["file_sha256"]:
            raise RuntimeError(
                f"Original source differs from manifest: {original}"
            )

        if sha256(raw_images[key]) != original_hash:
            raise RuntimeError(
                f"Frozen raw image differs from original: {original.name}"
            )

        verified.append({
            "filename": original.name,
            "raw_path": raw_images[key],
            "raw_sha256": original_hash,
            "frozen_path": frozen_images[key],
            "frozen_sha256": sha256(frozen_images[key]),
        })

    if set(raw_images) != names:
        raise ValueError(
            "Frozen raw image set differs from the detection manifest."
        )

    if set(frozen_images) != names:
        raise ValueError(
            "Frozen five-layer image set differs from the manifest."
        )

    return verified


def main():
    print(
        "Checking the manifest, source images, and checkpoint...",
        flush=True,
    )

    entries = validate_inputs()
    functions, extracted_text = extract_functions()

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    output = ROOT / f"five_layer_reproduction_audit_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    generated_folder = output / "regenerated_images"
    generated_folder.mkdir()

    (output / "extracted_benchmark_functions.txt").write_text(
        extracted_text,
        encoding="utf-8",
    )

    initial_hashes = {
        "source_script": sha256(SOURCE_SCRIPT),
        "source_manifest": sha256(SOURCE_MANIFEST),
        "configuration": sha256(CONFIGURATION),
        "checkpoint": sha256(CHECKPOINT),
    }

    metadata = {
        "status": "running",
        "expected_images": EXPECTED_IMAGES,
        "source_script": str(SOURCE_SCRIPT),
        "source_manifest": str(SOURCE_MANIFEST),
        "configuration": str(CONFIGURATION),
        "checkpoint": str(CHECKPOINT),
        "frozen_raw_images": str(FROZEN_RAW_IMAGES),
        "frozen_five_layer_images": str(FROZEN_FIVE_IMAGES),
        "input_hashes": initial_hashes,
        "audit_script_sha256": sha256(Path(__file__)),
        "generation_device": str(DEVICE),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "detector_evaluation_performed": False,
        "training_provenance_verified": False,
    }

    save_json(output / "experiment_metadata.json", metadata)

    print(
        f"Input checks passed: {len(entries)} images.\n"
        f"Output directory: {output}",
        flush=True,
    )

    model = functions["load_enhancer"](
        functions["FiveLayerEnhancementCNN"](),
        CHECKPOINT,
    )

    rows = []

    for index, entry in enumerate(entries, start=1):
        raw_path = entry["raw_path"]
        frozen_path = entry["frozen_path"]

        # Recheck files immediately before processing.
        if sha256(raw_path) != entry["raw_sha256"]:
            raise RuntimeError(f"Raw image changed: {raw_path}")
        if sha256(frozen_path) != entry["frozen_sha256"]:
            raise RuntimeError(f"Frozen image changed: {frozen_path}")

        raw = functions["read_image"](raw_path)
        generated = functions["enhance"](model, raw)

        generated_path = generated_folder / entry["filename"]
        functions["write_image"](generated_path, generated)

        frozen = cv2.imread(
            str(frozen_path), cv2.IMREAD_UNCHANGED
        )
        regenerated = cv2.imread(
            str(generated_path), cv2.IMREAD_UNCHANGED
        )

        if frozen is None or regenerated is None:
            raise RuntimeError(
                f"Image decoding failed: {entry['filename']}"
            )

        same_shape = frozen.shape == regenerated.shape
        same_dtype = frozen.dtype == regenerated.dtype

        regenerated_hash = sha256(generated_path)

        row = {
            "filename": entry["filename"],
            "raw_path": str(raw_path),
            "raw_sha256": entry["raw_sha256"],
            "frozen_path": str(frozen_path),
            "regenerated_path": str(generated_path),
            "frozen_sha256": entry["frozen_sha256"],
            "regenerated_sha256": regenerated_hash,
            "bytes_equal": (
                entry["frozen_sha256"] == regenerated_hash
            ),
            "frozen_shape": str(frozen.shape),
            "regenerated_shape": str(regenerated.shape),
            "same_shape": same_shape,
            "frozen_dtype": str(frozen.dtype),
            "regenerated_dtype": str(regenerated.dtype),
            "same_dtype": same_dtype,
            "pixels_equal": False,
            "mean_abs_pixel_difference": "",
            "max_abs_pixel_difference": "",
        }

        if same_shape:
            difference = np.abs(
                frozen.astype(np.float64)
                - regenerated.astype(np.float64)
            )

            row["pixels_equal"] = bool(
                same_dtype and np.array_equal(frozen, regenerated)
            )
            row["mean_abs_pixel_difference"] = float(
                difference.mean()
            )
            row["max_abs_pixel_difference"] = float(
                difference.max()
            )

        rows.append(row)

        if index % 50 == 0:
            save_csv(output / "image_audit_details.csv", rows)
            print(
                f"Compared {index}/{EXPECTED_IMAGES}; "
                f"byte-identical={sum(r['bytes_equal'] for r in rows)}, "
                f"pixel-identical={sum(r['pixels_equal'] for r in rows)}",
                flush=True,
            )

    final_hashes = {
        "source_script": sha256(SOURCE_SCRIPT),
        "source_manifest": sha256(SOURCE_MANIFEST),
        "configuration": sha256(CONFIGURATION),
        "checkpoint": sha256(CHECKPOINT),
    }

    if final_hashes != initial_hashes:
        raise RuntimeError(
            "A source/configuration/checkpoint file changed "
            "during the audit."
        )

    summary = {
        "matched_images": len(rows),
        "byte_identical": sum(r["bytes_equal"] for r in rows),
        "pixel_identical": sum(r["pixels_equal"] for r in rows),
        "shape_mismatches": sum(
            not r["same_shape"] for r in rows
        ),
        "dtype_mismatches": sum(
            not r["same_dtype"] for r in rows
        ),
        "not_pixel_identical": sum(
            not r["pixels_equal"] for r in rows
        ),
        "all_images_byte_identical": all(
            r["bytes_equal"] for r in rows
        ),
        "all_images_pixel_identical": all(
            r["pixels_equal"] for r in rows
        ),
        "detector_evaluation_performed": False,
        "interpretation": (
            "This audit checks regeneration of the frozen five-layer "
            "image artifacts with the current checkpoint and extracted "
            "benchmark functions. It does not recover training "
            "provenance or replay detector metrics."
        ),
    }

    save_csv(output / "image_audit_details.csv", rows)
    save_json(output / "image_audit_summary.json", summary)

    metadata["status"] = "complete"
    metadata["summary"] = summary
    save_json(output / "experiment_metadata.json", metadata)

    print("\nFIVE-LAYER REPRODUCTION AUDIT")
    print(json.dumps(summary, indent=2))
    print(f"\nReports and regenerated images saved to:\n{output}")


if __name__ == "__main__":
    main()