import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"C:\internnn")
RAW_DIR = ROOT / "URPC2020" / "images" / "test"
ALPHA0_DIR = ROOT / "alpha_sweep" / "alpha_0_00" / "images"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    result = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        key = path.name.casefold()
        if key in result:
            raise ValueError(
                "Ambiguous duplicate filename:\n"
                f"{result[key]}\n{path}\n"
                "Select a narrower image folder before comparing."
            )
        result[key] = path

    if not result:
        raise ValueError(f"No images found in: {folder}")

    return result


def main():
    raw = inventory(RAW_DIR)
    sweep = inventory(ALPHA0_DIR)

    print(f"Raw images found:    {len(raw)}")
    print(f"Alpha0 images found: {len(sweep)}")

    only_raw = sorted(set(raw) - set(sweep))
    only_sweep = sorted(set(sweep) - set(raw))

    if only_raw or only_sweep:
        print("\nExamples found only in raw:", only_raw[:10])
        print("Examples found only in alpha0:", only_sweep[:10])
        raise ValueError(
            "Filename sets differ. No pixel comparison performed. "
            "Do not guess filename remappings."
        )

    if len(raw) != 800:
        raise ValueError(
            f"Expected 800 matched images; found {len(raw)}."
        )

    output = ROOT / "revision_audits" / (
        "raw_vs_alpha0_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    for i, key in enumerate(sorted(raw), start=1):
        a = raw[key]
        b = sweep[key]

        x = cv2.imread(str(a), cv2.IMREAD_UNCHANGED)
        y = cv2.imread(str(b), cv2.IMREAD_UNCHANGED)

        if x is None or y is None:
            raise ValueError(f"Image decoding failed:\n{a}\n{b}")

        hash_a = sha256(a)
        hash_b = sha256(b)
        same_shape = x.shape == y.shape
        same_dtype = x.dtype == y.dtype

        row = {
            "filename": a.name,
            "raw_path": str(a),
            "alpha0_path": str(b),
            "raw_sha256": hash_a,
            "alpha0_sha256": hash_b,
            "bytes_equal": hash_a == hash_b,
            "raw_shape": str(x.shape),
            "alpha0_shape": str(y.shape),
            "raw_dtype": str(x.dtype),
            "alpha0_dtype": str(y.dtype),
            "pixels_equal": False,
            "mean_abs_difference": "",
            "max_abs_difference": "",
        }

        if same_shape:
            difference = np.abs(
                x.astype(np.float64) - y.astype(np.float64)
            )
            row["pixels_equal"] = bool(
                same_dtype and np.array_equal(x, y)
            )
            row["mean_abs_difference"] = float(difference.mean())
            row["max_abs_difference"] = float(difference.max())

        rows.append(row)

        if i % 100 == 0:
            print(f"Compared {i}/800 image pairs", flush=True)

    with open(output / "image_comparison.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "raw_folder": str(RAW_DIR),
        "alpha0_folder": str(ALPHA0_DIR),
        "matching_method": "unique case-insensitive full filename",
        "matched_images": len(rows),
        "byte_identical": sum(r["bytes_equal"] for r in rows),
        "pixel_identical": sum(r["pixels_equal"] for r in rows),
        "shape_mismatches": sum(
            r["raw_shape"] != r["alpha0_shape"] for r in rows
        ),
        "dtype_mismatches": sum(
            r["raw_dtype"] != r["alpha0_dtype"] for r in rows
        ),
        "not_pixel_identical": sum(
            not r["pixels_equal"] for r in rows
        ),
        "labels_audited": False,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nRAW VS ALPHA0 — CORRECTED AUDIT")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved reports: {output}")


if __name__ == "__main__":
    main()