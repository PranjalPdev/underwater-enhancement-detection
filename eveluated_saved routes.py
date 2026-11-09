import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO


# =====================================================================
# EXISTING COMPLETED RUN
# =====================================================================
RUN_DIR = Path(
    r"C:\internnn\temp_eval"
    r"\proposed_pipeline_20261019T203221_524771Z"
)

POOLED_EVAL_DIR = RUN_DIR / (
    "detection_evaluation_20261020T042244_360547Z"
)

ROUTING_MANIFEST = RUN_DIR / "routing_manifest.csv"
DETECTION_MANIFEST = RUN_DIR / "detection_manifest.json"
POOLED_CONFIG = POOLED_EVAL_DIR / "evaluation_configuration.json"

YOLO_WEIGHTS = Path(
    r"C:\Users\pppan\runs\detect"
    r"\urpc2020_yolov8n-9\weights\best.pt"
)

RAW_DATASET = POOLED_EVAL_DIR / "datasets" / "raw"
DYNAMIC_DATASET = (
    POOLED_EVAL_DIR / "datasets" / "dynamic_efficientnet"
)

CLASS_NAMES = ["holothurian", "echinus", "scallop", "starfish"]
ROUTES = ["good", "low_light", "turbid"]

EXPECTED_COUNTS = {
    "good": 223,
    "low_light": 399,
    "turbid": 178,
}

RUN_ID = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%S_%fZ"
)

# Only a new results directory is created.
OUTPUT_DIR = POOLED_EVAL_DIR / f"routewise_{RUN_ID}"


# =====================================================================
# UTILITIES
# =====================================================================
def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_key(path):
    return str(Path(path).resolve()).casefold()


# =====================================================================
# VERIFY THAT THE SAVED DATASETS MATCH THE MANIFESTS
# =====================================================================
def load_and_verify():
    for path in [
        ROUTING_MANIFEST,
        DETECTION_MANIFEST,
        POOLED_CONFIG,
        YOLO_WEIGHTS,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    for dataset in [RAW_DATASET, DYNAMIC_DATASET]:
        for subfolder in ["images", "labels"]:
            if not (dataset / subfolder).is_dir():
                raise FileNotFoundError(dataset / subfolder)

    config = json.loads(
        POOLED_CONFIG.read_text(encoding="utf-8")
    )

    if file_sha256(YOLO_WEIGHTS) != config["yolo_checkpoint_sha256"]:
        raise RuntimeError(
            "YOLO checkpoint differs from the pooled evaluation."
        )

    recorded_version = config.get("ultralytics_version")
    if recorded_version != ultralytics.__version__:
        raise RuntimeError(
            "Ultralytics version differs from the pooled run.\n"
            f"Pooled run: {recorded_version}\n"
            f"Current: {ultralytics.__version__}\n"
            "Use the same environment before comparing results."
        )

    detection = json.loads(
        DETECTION_MANIFEST.read_text(encoding="utf-8")
    )
    routes = pd.read_csv(
        ROUTING_MANIFEST,
        keep_default_na=False,
    ).to_dict("records")

    detection_lookup = {}
    for record in detection:
        key = path_key(record["path"])
        if key in detection_lookup:
            raise ValueError("Duplicate source in detection manifest.")
        detection_lookup[key] = record

    if len(routes) != len(detection_lookup):
        raise ValueError("Routing and detection manifest sizes differ.")

    records = []
    seen_sources = set()
    seen_names = set()

    for row in routes:
        source = Path(row["source"])
        key = path_key(source)

        if key in seen_sources:
            raise ValueError(f"Repeated routing source: {source}")
        seen_sources.add(key)

        if key not in detection_lookup:
            raise ValueError(f"Unknown routing source: {source}")

        if row["route"] not in ROUTES:
            raise ValueError(f"Unknown route: {row['route']}")

        if source.name.casefold() in seen_names:
            raise ValueError(f"Repeated filename: {source.name}")
        seen_names.add(source.name.casefold())

        raw_image = RAW_DATASET / "images" / source.name
        dynamic_image = DYNAMIC_DATASET / "images" / source.name

        raw_label = RAW_DATASET / "labels" / f"{source.stem}.txt"
        dynamic_label = (
            DYNAMIC_DATASET / "labels" / f"{source.stem}.txt"
        )

        for path in [
            raw_image,
            dynamic_image,
            raw_label,
            dynamic_label,
        ]:
            if not path.is_file():
                raise FileNotFoundError(path)

        # Check the saved raw copy against the audited source manifest.
        expected_hash = detection_lookup[key]["file_sha256"]
        if file_sha256(raw_image) != expected_hash:
            raise RuntimeError(
                f"Saved raw image differs from manifest: {raw_image}"
            )

        # Check saved dynamic copy against the original routing output.
        routed_output = Path(row["output"])
        if not routed_output.is_file():
            raise FileNotFoundError(routed_output)

        if file_sha256(dynamic_image) != file_sha256(routed_output):
            raise RuntimeError(
                f"Saved dynamic image differs from routing output: "
                f"{dynamic_image}"
            )

        # Both evaluations must use the same cleaned annotations.
        if raw_label.read_bytes() != dynamic_label.read_bytes():
            raise RuntimeError(
                f"Raw/dynamic annotations differ: {source.stem}"
            )

        # Bypass must be unchanged.
        if row["route"] == "good":
            if file_sha256(raw_image) != file_sha256(dynamic_image):
                raise RuntimeError(
                    f"Bypassed image is not identical: {source.name}"
                )

        records.append({
            "filename": source.name,
            "route": row["route"],
            "raw_image": str(raw_image.resolve()),
            "dynamic_image": str(dynamic_image.resolve()),
            "raw_label": str(raw_label.resolve()),
            "dynamic_label": str(dynamic_label.resolve()),
            "raw_label_sha256": file_sha256(raw_label),
            "dynamic_image_sha256": file_sha256(dynamic_image),
        })

    if seen_sources != set(detection_lookup):
        raise ValueError("Not all detection images have routing records.")

    counts = Counter(record["route"] for record in records)
    actual_counts = {route: counts[route] for route in ROUTES}

    if actual_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Unexpected route counts.\n"
            f"Expected: {EXPECTED_COUNTS}\n"
            f"Found: {actual_counts}"
        )

    return records, config


# =====================================================================
# DATASET LISTS REFERENCE EXISTING IMAGES AND CLEANED LABELS
# =====================================================================
def create_subset_yaml(records, route, variant):
    selected = [r for r in records if r["route"] == route]

    list_path = OUTPUT_DIR / f"{route}_{variant}_images.txt"

    list_path.write_text(
        "\n".join(
            Path(record[f"{variant}_image"]).as_posix()
            for record in selected
        ) + "\n",
        encoding="utf-8",
    )

    yaml_path = OUTPUT_DIR / f"{route}_{variant}.yaml"

    # JSON syntax is valid YAML.
    # Existing images/... paths allow Ultralytics to locate
    # the corresponding existing labels/... files.
    write_json(yaml_path, {
        "path": OUTPUT_DIR.resolve().as_posix(),
        "train": list_path.resolve().as_posix(),
        "val": list_path.resolve().as_posix(),
        "test": list_path.resolve().as_posix(),
        "names": {
            index: name
            for index, name in enumerate(CLASS_NAMES)
        },
    })

    return yaml_path, len(selected)


# =====================================================================
# EVALUATION
# =====================================================================
def evaluate_subset(records, config, route, variant):
    yaml_path, image_count = create_subset_yaml(
        records, route, variant
    )
    name = f"{route}_{variant}"

    model = YOLO(str(YOLO_WEIGHTS))
    actual_names = [
        model.names[index]
        for index in range(len(model.names))
    ]

    if actual_names != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected detector classes: {actual_names}"
        )

    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=int(config["imgsz"]),
        rect=bool(config["rect"]),
        batch=int(config["batch"]),
        device="cpu",
        workers=0,
        conf=float(config["conf"]),
        iou=float(config["iou"]),
        max_det=int(config["max_det"]),
        half=False,
        augment=False,
        plots=True,
        save_json=True,
        project=str(OUTPUT_DIR / "ultralytics_results"),
        name=name,
        exist_ok=False,
        verbose=True,
    )

    class_rows = []
    for position, class_id in enumerate(metrics.box.ap_class_index):
        class_id = int(class_id)

        class_rows.append({
            "route": route,
            "variant": variant,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "map50": float(metrics.box.ap50[position]),
            "map50_95": float(metrics.box.ap[position]),
        })

    pd.DataFrame(class_rows).to_csv(
        OUTPUT_DIR / f"{name}_per_class.csv",
        index=False,
    )

    return {
        "route": route,
        "variant": variant,
        "images": image_count,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main():
    records, config = load_and_verify()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "verified_routewise_manifest.csv",
        index=False,
    )

    write_json(OUTPUT_DIR / "configuration.json", {
        "pooled_evaluation_directory": str(POOLED_EVAL_DIR),
        "source_configuration": config,
        "torch_version_current": torch.__version__,
        "ultralytics_version_current": ultralytics.__version__,
        "route_counts": EXPECTED_COUNTS,
        "note": (
            "Existing images and cleaned labels are reused. "
            "No classifier training or enhancement generation occurs."
        ),
    })

    individual_results = []
    comparisons = []

    for route in ROUTES:
        scores = {}

        for variant in ["raw", "dynamic"]:
            print("\n" + "=" * 72)
            print(f"EVALUATING ROUTE: {route} | VARIANT: {variant}")
            print("=" * 72)

            result = evaluate_subset(
                records, config, route, variant
            )
            individual_results.append(result)
            scores[variant] = result

            # Save progress after every completed evaluation.
            pd.DataFrame(individual_results).to_csv(
                OUTPUT_DIR / "individual_results.csv",
                index=False,
            )
            write_json(
                OUTPUT_DIR / "individual_results.json",
                individual_results,
            )

        comparison = {
            "route": route,
            "images": scores["raw"]["images"],
            "raw_map50": scores["raw"]["map50"],
            "pipeline_map50": scores["dynamic"]["map50"],
            "delta_map50": (
                scores["dynamic"]["map50"]
                - scores["raw"]["map50"]
            ),
            "raw_map50_95": scores["raw"]["map50_95"],
            "pipeline_map50_95": scores["dynamic"]["map50_95"],
            "delta_map50_95": (
                scores["dynamic"]["map50_95"]
                - scores["raw"]["map50_95"]
            ),
        }
        comparisons.append(comparison)

        pd.DataFrame(comparisons).to_csv(
            OUTPUT_DIR / "routewise_comparison.csv",
            index=False,
        )
        write_json(
            OUTPUT_DIR / "routewise_comparison.json",
            comparisons,
        )

    print("\n" + "=" * 88)
    print("FINAL ROUTE-WISE DETECTION COMPARISON")
    print("=" * 88)

    frame = pd.DataFrame(comparisons)
    print(frame[
        [
            "route",
            "images",
            "raw_map50",
            "pipeline_map50",
            "delta_map50",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print("\nResults saved to:")
    print(OUTPUT_DIR)
    print(
        "\nNo full-pipeline latency was measured. "
        "Route-wise AP values must not be averaged to "
        "reconstruct the pooled detection metric."
    )


if __name__ == "__main__":
    main()