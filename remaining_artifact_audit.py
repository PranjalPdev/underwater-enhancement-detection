import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(r"C:\internnn")
RAW_IMAGES = ROOT / "URPC2020" / "images" / "test"
ALPHA0_LABELS = ROOT / "alpha_sweep" / "alpha_0_00" / "labels"
ALPHA35_IMAGES = ROOT / "alpha_sweep" / "alpha_0_35" / "images"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ask_path(prompt, kind="directory", optional=False):
    while True:
        value = input(f"\n{prompt}\n> ").strip().strip('"').strip("'")

        if not value and optional:
            return None

        path = Path(value)

        if kind == "directory" and path.is_dir():
            return path
        if kind == "file" and path.is_file():
            return path

        print(f"Not an existing {kind}: {value}")


def inventory(folder, extensions):
    result = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue

        key = path.name.casefold()
        if key in result:
            raise ValueError(
                "Ambiguous duplicate filename. Select a narrower folder:\n"
                f"{result[key]}\n{path}"
            )
        result[key] = path

    if not result:
        raise ValueError(f"No matching files found in {folder}")

    return result


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=str),
        encoding="utf-8",
    )


def read_labels(path):
    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue

        values = [float(x) for x in line.split()]

        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError(
                f"Invalid detection label at {path}:{line_number}"
            )

        class_id, x, y, w, h = values
        if class_id != int(class_id) or not 0 <= class_id <= 3:
            raise ValueError(
                f"Invalid class ID at {path}:{line_number}"
            )
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Non-positive box size at {path}:{line_number}. "
                "This may be an uncleaned label folder."
            )

        records.append(tuple(values))

    return records


def audit_labels(output, raw_images):
    print("\nLABEL AUDIT")
    original = ask_path(
        "Paste the ORIGINAL CLEANED benchmark label-folder path."
    )

    sweep = ALPHA0_LABELS
    if not sweep.is_dir():
        sweep = ask_path("Paste the alpha=0 sweep label-folder path.")

    if original.resolve() == sweep.resolve():
        raise ValueError("Select two different label folders.")

    left = inventory(original, {".txt"})
    right = inventory(sweep, {".txt"})

    expected = [
        Path(name).with_suffix(".txt").name.casefold()
        for name in raw_images
    ]

    if len(expected) != len(set(expected)):
        raise ValueError("Image names map to duplicate label filenames.")

    rows = []
    total_left = 0
    total_right = 0

    for name in expected:
        a = left.get(name)
        b = right.get(name)

        row = {
            "filename": name,
            "original_path": str(a) if a else "",
            "alpha0_path": str(b) if b else "",
            "original_sha256": "",
            "alpha0_sha256": "",
            "bytes_equal": False,
            "original_boxes": "",
            "alpha0_boxes": "",
            "status": "",
            "error": "",
        }

        if a is None or b is None:
            row["status"] = "missing_label_file"
            rows.append(row)
            continue

        row["original_sha256"] = sha256(a)
        row["alpha0_sha256"] = sha256(b)
        row["bytes_equal"] = (
            row["original_sha256"] == row["alpha0_sha256"]
        )

        try:
            x = read_labels(a)
            y = read_labels(b)
            row["original_boxes"] = len(x)
            row["alpha0_boxes"] = len(y)
            total_left += len(x)
            total_right += len(y)

            if x == y:
                row["status"] = "exact_numeric_same_order"
            elif sorted(x) == sorted(y):
                row["status"] = "same_annotations_different_order"
            else:
                row["status"] = "different_annotations"
        except ValueError as exc:
            row["status"] = "invalid_label"
            row["error"] = str(exc)

        rows.append(row)

    write_csv(output / "raw_vs_alpha0_labels.csv", rows)

    summary = {
        "original_label_folder": str(original),
        "alpha0_label_folder": str(sweep),
        "expected_label_files": len(expected),
        "byte_identical_pairs": sum(r["bytes_equal"] for r in rows),
        "exact_numeric_same_order": sum(
            r["status"] == "exact_numeric_same_order" for r in rows
        ),
        "same_annotations_different_order": sum(
            r["status"] == "same_annotations_different_order"
            for r in rows
        ),
        "different_annotations": sum(
            r["status"] == "different_annotations" for r in rows
        ),
        "missing_label_pairs": sum(
            r["status"] == "missing_label_file" for r in rows
        ),
        "invalid_label_pairs": sum(
            r["status"] == "invalid_label" for r in rows
        ),
        "original_box_count_successfully_parsed_pairs": total_left,
        "alpha0_box_count_successfully_parsed_pairs": total_right,
        "all_pairs_exact_numeric_same_order": all(
            r["status"] == "exact_numeric_same_order" for r in rows
        ),
    }

    save_json(output / "label_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def audit_residual_images(output, raw_images):
    print("\nFROZEN VERSUS REGENERATED RESIDUAL IMAGE AUDIT")

    original = ask_path(
        "Paste the ORIGINAL FIXED-RESIDUAL benchmark image-folder path.\n"
        "Do not select any folder inside alpha_sweep.",
        optional=True,
    )
    if original is None:
        return {"status": "skipped_original_folder_not_identified"}

    sweep_root = (ROOT / "alpha_sweep").resolve()
    if original.resolve().is_relative_to(sweep_root):
        raise ValueError(
            "The original residual images must come from the frozen "
            "benchmark, not alpha_sweep."
        )

    sweep = ALPHA35_IMAGES
    if not sweep.is_dir():
        sweep = ask_path(
            "Paste the regenerated alpha=0.35 image-folder path."
        )

    left = inventory(original, IMAGE_EXTENSIONS)
    right = inventory(sweep, IMAGE_EXTENSIONS)
    names = sorted(raw_images)

    rows = []
    for i, name in enumerate(names, start=1):
        a = left.get(name)
        b = right.get(name)

        row = {
            "filename": name,
            "frozen_path": str(a) if a else "",
            "regenerated_path": str(b) if b else "",
            "frozen_sha256": "",
            "regenerated_sha256": "",
            "bytes_equal": False,
            "pixels_equal": False,
            "frozen_shape": "",
            "regenerated_shape": "",
            "frozen_dtype": "",
            "regenerated_dtype": "",
            "mean_abs_pixel_difference": "",
            "max_abs_pixel_difference": "",
            "status": "",
        }

        if a is None or b is None:
            row["status"] = "missing_image"
            rows.append(row)
            continue

        row["frozen_sha256"] = sha256(a)
        row["regenerated_sha256"] = sha256(b)
        row["bytes_equal"] = (
            row["frozen_sha256"] == row["regenerated_sha256"]
        )

        x = cv2.imread(str(a), cv2.IMREAD_UNCHANGED)
        y = cv2.imread(str(b), cv2.IMREAD_UNCHANGED)

        if x is None or y is None:
            row["status"] = "decode_failure"
        else:
            row["frozen_shape"] = str(x.shape)
            row["regenerated_shape"] = str(y.shape)
            row["frozen_dtype"] = str(x.dtype)
            row["regenerated_dtype"] = str(y.dtype)

            if x.shape != y.shape:
                row["status"] = "shape_mismatch"
            else:
                difference = np.abs(
                    x.astype(np.float64) - y.astype(np.float64)
                )
                equal = (
                    x.dtype == y.dtype and np.array_equal(x, y)
                )
                row["pixels_equal"] = bool(equal)
                row["mean_abs_pixel_difference"] = float(
                    difference.mean()
                )
                row["max_abs_pixel_difference"] = float(
                    difference.max()
                )
                row["status"] = (
                    "identical_pixels" if equal else "different_pixels"
                )

        rows.append(row)

        if i % 100 == 0:
            print(f"Compared residual images: {i}/{len(names)}",
                  flush=True)

    write_csv(output / "residual_image_comparison.csv", rows)

    summary = {
        "frozen_folder": str(original),
        "regenerated_folder": str(sweep),
        "expected_images": len(names),
        "byte_identical": sum(r["bytes_equal"] for r in rows),
        "pixel_identical": sum(r["pixels_equal"] for r in rows),
        "different_pixels": sum(
            r["status"] == "different_pixels" for r in rows
        ),
        "shape_mismatches": sum(
            r["status"] == "shape_mismatch" for r in rows
        ),
        "missing_image_pairs": sum(
            r["status"] == "missing_image" for r in rows
        ),
        "decode_failures": sum(
            r["status"] == "decode_failure" for r in rows
        ),
    }

    save_json(output / "residual_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def read_record(path):
    text = path.read_text(encoding="utf-8-sig")
    data = (
        json.loads(text)
        if path.suffix.lower() == ".json"
        else yaml.safe_load(text)
    )
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")

    # Accept a direct argument mapping or an explicitly named wrapper.
    if isinstance(data.get("args"), dict):
        return data["args"]
    if isinstance(data.get("validation_args"), dict):
        return data["validation_args"]
    return data


def audit_evaluation_records(output):
    print(
        "\nOPTIONAL EVALUATION-ARGUMENT AUDIT\n"
        "Select saved VALIDATION arguments, not training arguments "
        "or dataset data.yaml.\n"
        "Press Enter to skip if those records are unavailable."
    )

    a = ask_path(
        "Original raw benchmark validation-argument YAML/JSON:",
        kind="file", optional=True,
    )
    if a is None:
        return {"status": "not_verified_no_argument_records"}

    b = ask_path(
        "Alpha=0 sweep validation-argument YAML/JSON:",
        kind="file", optional=True,
    )
    if b is None:
        return {"status": "not_verified_second_record_missing"}

    left = read_record(a)
    right = read_record(b)

    keys = [
        "task", "mode", "model", "data", "split",
        "imgsz", "batch", "rect", "conf", "iou", "max_det",
        "device", "half", "augment", "agnostic_nms", "classes",
        "single_cls", "dnn", "workers", "seed", "deterministic",
        "fraction", "save_json",
    ]

    rows = []
    for key in keys:
        present_left = key in left
        present_right = key in right

        rows.append({
            "argument": key,
            "original_present": present_left,
            "alpha0_present": present_right,
            "original_value": json.dumps(
                left.get(key), default=str, sort_keys=True
            ),
            "alpha0_value": json.dumps(
                right.get(key), default=str, sort_keys=True
            ),
            "status": (
                "missing_from_one_or_both"
                if not (present_left and present_right)
                else "equal" if left[key] == right[key]
                else "different"
            ),
        })

    write_csv(output / "evaluation_argument_comparison.csv", rows)

    summary = {
        "original_record": str(a),
        "alpha0_record": str(b),
        "original_record_sha256": sha256(a),
        "alpha0_record_sha256": sha256(b),
        "different_arguments": [
            r["argument"] for r in rows if r["status"] == "different"
        ],
        "unverified_arguments": [
            r["argument"] for r in rows
            if r["status"] == "missing_from_one_or_both"
        ],
        "warning": (
            "Argument records are documentary evidence only. "
            "Different data paths may reference equivalent artifacts. "
            "Missing fields are not assumed to have equal defaults."
        ),
    }

    print(
        "\nOptional detector-checkpoint hash comparison. "
        "Press Enter to skip."
    )
    weight_a = ask_path(
        "Detector checkpoint used for original benchmark:",
        kind="file", optional=True,
    )
    if weight_a:
        weight_b = ask_path(
            "Detector checkpoint used for alpha=0 sweep:",
            kind="file", optional=True,
        )
        if weight_b:
            hash_a = sha256(weight_a)
            hash_b = sha256(weight_b)
            summary["checkpoint_comparison"] = {
                "original_path": str(weight_a),
                "alpha0_path": str(weight_b),
                "original_sha256": hash_a,
                "alpha0_sha256": hash_b,
                "bytes_equal": hash_a == hash_b,
                "warning": (
                    "Hashes describe the files available now; "
                    "historical use requires corresponding records."
                ),
            }

    save_json(output / "evaluation_record_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    output = ROOT / "remaining_artifact_audits" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    raw_images = inventory(RAW_IMAGES, IMAGE_EXTENSIONS)
    if len(raw_images) != 800:
        raise ValueError(
            f"Expected 800 raw images, found {len(raw_images)}."
        )

    summary = {
        "environment": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        }
    }

    tasks = [
        ("labels", lambda: audit_labels(output, raw_images)),
        ("residual_images",
         lambda: audit_residual_images(output, raw_images)),
        ("evaluation_records",
         lambda: audit_evaluation_records(output)),
    ]

    for name, task in tasks:
        try:
            summary[name] = task()
        except Exception as exc:
            summary[name] = {
                "status": "failed",
                "error": str(exc),
            }
            print(f"\n{name} audit stopped: {exc}")

        save_json(output / "combined_summary.json", summary)

    print("\nFINAL AUDIT SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nReports saved to: {output}")


if __name__ == "__main__":
    main()