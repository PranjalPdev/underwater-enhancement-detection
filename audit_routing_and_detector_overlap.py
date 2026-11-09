import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(r"C:\internnn")
RUN_DIR = (
    ROOT / "temp_eval"
    / "proposed_pipeline_20261019T203221_524771Z"
)

ROUTING_MANIFEST = RUN_DIR / "routing_manifest.csv"
DETECTION_MANIFEST = RUN_DIR / "detection_manifest.json"
CONFIGURATION = RUN_DIR / "configuration.json"

TRAIN_ARGS = Path(
    r"C:\Users\pppan\runs\detect"
    r"\urpc2020_yolov8n-9\args.yaml"
)
DATA_YAML = ROOT / "URPC2020" / "data.yaml"

EXPECTED_IMAGES = 800
EXPECTED_COUNTS = {
    "good": 223,
    "low_light": 399,
    "turbid": 178,
}

ROUTE_NAMES = ["good", "low_light", "turbid"]
PROBABILITY_COLUMNS = ["p_good", "p_low_light", "p_turbid"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp", ".dng",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(path):
    return str(Path(path).resolve()).casefold()


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )

def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found: {folder}")
    return paths


def pixel_hash(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Image decode failed: {path}")

    h = hashlib.sha256()
    h.update(str(image.shape).encode("utf-8"))
    h.update(str(image.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def fingerprint(path):
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256(path),
        "decoded_bgr_sha256": pixel_hash(path),
    }


def audit_routes(output):
    with open(
        ROUTING_MANIFEST, newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "source", "output", "route", *PROBABILITY_COLUMNS
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Routing manifest missing columns: {sorted(missing)}"
            )

        rows = list(reader)

    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected 800 routing records, found {len(rows)}."
        )

    manifest = json.loads(
        DETECTION_MANIFEST.read_text(encoding="utf-8-sig")
    )
    if not isinstance(manifest, list) or len(manifest) != EXPECTED_IMAGES:
        raise ValueError("Expected an 800-record detection manifest.")

    expected_sources = {}
    for record in manifest:
        key = canonical(record["path"])
        if key in expected_sources:
            raise ValueError("Duplicate detection-manifest source.")
        expected_sources[key] = record

    observed_sources = [canonical(row["source"]) for row in rows]

    if len(set(observed_sources)) != EXPECTED_IMAGES:
        raise ValueError("Duplicate routing sources.")

    if set(observed_sources) != set(expected_sources):
        raise ValueError(
            "Routing and detection manifests contain different images."
        )

    details = []
    counts = Counter()
    tie_count = 0

    for row in rows:
        source = Path(row["source"])
        processed = Path(row["output"])
        key = canonical(source)

        if not source.is_file():
            raise FileNotFoundError(source)
        if not processed.is_file():
            raise FileNotFoundError(processed)

        if sha256(source) != expected_sources[key]["file_sha256"]:
            raise RuntimeError(
                f"Source no longer matches detection manifest: {source}"
            )

        probabilities = np.asarray(
            [float(row[column]) for column in PROBABILITY_COLUMNS],
            dtype=np.float64,
        )

        if not np.isfinite(probabilities).all():
            raise ValueError(f"Nonfinite probabilities: {source}")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Probability outside [0,1]: {source}")
        if not np.isclose(
            probabilities.sum(), 1.0, atol=1e-5, rtol=0
        ):
            raise ValueError(f"Probabilities do not sum to one: {source}")

        expected_route = ROUTE_NAMES[int(np.argmax(probabilities))]
        saved_route = row["route"]

        if saved_route not in ROUTE_NAMES:
            raise ValueError(f"Unknown saved route: {saved_route}")

        tied = int(np.sum(probabilities == probabilities.max())) > 1
        tie_count += int(tied)
        counts[saved_route] += 1

        details.append({
            "source": str(source),
            "saved_route": saved_route,
            "argmax_route": expected_route,
            "matches_argmax": saved_route == expected_route,
            "exact_maximum_tie": tied,
            **{
                name: float(value)
                for name, value in zip(
                    PROBABILITY_COLUMNS, probabilities
                )
            },
        })

    config = json.loads(
        CONFIGURATION.read_text(encoding="utf-8-sig")
    )

    mismatches = [
        row for row in details if not row["matches_argmax"]
    ]

    save_csv(
        output / "routing_assignment_details.csv",
        details,
        list(details[0]),
    )

    summary = {
        "images": len(rows),
        "route_counts": dict(counts),
        "counts_match_reported": dict(counts) == EXPECTED_COUNTS,
        "argmax_matches": len(rows) - len(mismatches),
        "argmax_mismatches": len(mismatches),
        "exact_maximum_ties": tie_count,
        "recorded_routing_policy": config.get("routing_policy"),
        "interpretation": (
            "Checks saved assignments against saved probability "
            "columns in the stated good/low_light/turbid order. "
            "Does not independently rerun the classifier or prove "
            "which historical script version generated the records."
        ),
    }
    save_json(output / "routing_summary.json", summary)
    return summary, manifest


def configured_split_directory(data, split):
    value = data.get(split)

    # These supplied records use one relative directory per split.
    # Stop rather than guess if the configuration changes format.
    if not isinstance(value, str):
        raise ValueError(
            f"{split}: expected one directory string, found {value!r}"
        )

    base = Path(data.get("path", DATA_YAML.parent))
    if not base.is_absolute():
        base = DATA_YAML.parent / base

    path = Path(value)
    if not path.is_absolute():
        path = base / path

    if not path.is_dir():
        raise ValueError(
            f"{split}: expected an existing image directory: {path}"
        )
    return path.resolve()


def audit_overlap(output, manifest):
    args = yaml.safe_load(
        TRAIN_ARGS.read_text(encoding="utf-8-sig")
    )
    data = yaml.safe_load(
        DATA_YAML.read_text(encoding="utf-8-sig")
    )

    if canonical(args["data"]) != canonical(DATA_YAML):
        raise RuntimeError(
            "Training args refer to a different dataset YAML."
        )

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names != ["holothurian", "echinus", "scallop", "starfish"]:
        raise ValueError("Unexpected dataset class mapping.")

    split_paths = {
        split: configured_split_directory(data, split)
        for split in ["train", "val", "test"]
    }

    evaluation = []
    for index, record in enumerate(manifest, start=1):
        path = Path(record["path"])
        item = fingerprint(path)

        if item["file_sha256"] != record["file_sha256"]:
            raise RuntimeError(f"Evaluation source changed: {path}")

        evaluation.append(item)

        if index % 200 == 0:
            print(
                f"Fingerprinted evaluation: {index}/800",
                flush=True,
            )

    configured_test = {
        canonical(p) for p in image_files(split_paths["test"])
    }
    manifest_paths = {canonical(r["path"]) for r in manifest}

    if configured_test != manifest_paths:
        raise RuntimeError(
            "Current configured test file set differs from "
            "the 800-image evaluation manifest."
        )

    save_csv(
        output / "evaluation_image_fingerprints.csv",
        evaluation,
        list(evaluation[0]),
    )

    evaluation_byte_index = defaultdict(list)
    evaluation_pixel_index = defaultdict(list)

    for item in evaluation:
        evaluation_byte_index[item["file_sha256"]].append(item["path"])
        evaluation_pixel_index[
            item["decoded_bgr_sha256"]
        ].append(item["path"])

    all_matches = []
    split_summaries = {}

    for split in ["train", "val"]:
        files = image_files(split_paths[split])
        fingerprints = []
        matched_eval_byte = set()
        matched_eval_pixel = set()
        matched_split_files = set()

        print(
            f"\nChecking detector {split}: {len(files)} images",
            flush=True,
        )

        for index, path in enumerate(files, start=1):
            item = fingerprint(path)
            fingerprints.append(item)

            byte_matches = set(
                evaluation_byte_index.get(item["file_sha256"], [])
            )
            pixel_matches = set(
                evaluation_pixel_index.get(
                    item["decoded_bgr_sha256"], []
                )
            )

            matched_eval_byte.update(byte_matches)
            matched_eval_pixel.update(pixel_matches)

            for evaluation_path in sorted(byte_matches | pixel_matches):
                matched_split_files.add(item["path"])
                all_matches.append({
                    "detector_split": split,
                    "split_image": item["path"],
                    "evaluation_image": evaluation_path,
                    "byte_identical": evaluation_path in byte_matches,
                    "decoded_pixel_identical": (
                        evaluation_path in pixel_matches
                    ),
                })

            if index % 200 == 0 or index == len(files):
                print(
                    f"{split}: {index}/{len(files)} checked",
                    flush=True,
                )

        save_csv(
            output / f"detector_{split}_image_fingerprints.csv",
            fingerprints,
            list(fingerprints[0]),
        )

        split_summaries[split] = {
            "directory": str(split_paths[split]),
            "images_checked": len(files),
            "evaluation_images_with_byte_matches": len(
                matched_eval_byte
            ),
            "evaluation_images_with_pixel_matches": len(
                matched_eval_pixel
            ),
            "split_images_matching_evaluation": len(
                matched_split_files
            ),
        }

    save_csv(
        output / "overlap_pairs.csv",
        all_matches,
        [
            "detector_split",
            "split_image",
            "evaluation_image",
            "byte_identical",
            "decoded_pixel_identical",
        ],
    )

    summary = {
        "evaluation_images": len(evaluation),
        "current_test_set_matches_manifest": True,
        "detector_splits": split_summaries,
        "matching_pairs": len(all_matches),
        "no_exact_matches_in_current_train_or_val": len(all_matches) == 0,
        "historical_training_inventory_verified": False,
        "scope": (
            "Audits files currently present in the train and val "
            "directories identified by the supplied training args "
            "and current data YAML. Historical directory contents, "
            "additional training runs, transformed near-duplicates, "
            "and scene/sequence overlap are not established."
        ),
    }
    save_json(output / "detector_overlap_summary.json", summary)
    return summary


def main():
    required = [
        ROUTING_MANIFEST,
        DETECTION_MANIFEST,
        CONFIGURATION,
        TRAIN_ARGS,
        DATA_YAML,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    output = ROOT / (
        "routing_detector_overlap_audit_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)

    initial_hashes = {str(p): sha256(p) for p in required}

    print("Auditing saved routing assignments...", flush=True)
    routes, manifest = audit_routes(output)

    print("\nROUTING SUMMARY")
    print(json.dumps(routes, indent=2))

    overlap = audit_overlap(output, manifest)

    final_hashes = {str(p): sha256(p) for p in required}
    if initial_hashes != final_hashes:
        raise RuntimeError(
            "A manifest or configuration changed during the audit."
        )

    summary = {
        "routing": routes,
        "detector_overlap": overlap,
        "input_file_hashes": initial_hashes,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "audit_script_sha256": sha256(Path(__file__)),
    }

    save_json(output / "combined_summary.json", summary)

    print("\nFINAL ROUTING AND DETECTOR-OVERLAP AUDIT")
    print(json.dumps(summary, indent=2))
    print(f"\nReports saved to:\n{output}")


if __name__ == "__main__":
    main()