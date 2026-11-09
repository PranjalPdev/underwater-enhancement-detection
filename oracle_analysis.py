"""
Oracle best-of-three analysis for underwater detection.

Selects, for every evaluation image, the processing branch with the
highest test-label-informed detection utility:

    utility(image, branch) = TP - FP

TP and FP are counted at a fixed confidence threshold and IoU
threshold using the native per-image validator statistics.

Then recomputes pooled mAP50 from the selected branch's detections.

This is an optimistic, test-label-informed diagnostic bound.
It is NOT a deployable routing policy.

Usage:
    python oracle_analysis.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

NATIVE_DIR = Path(
    r"C:\internnn\temp_eval\proposed_pipeline_20261019T203221_524771Z"
    r"\detection_evaluation_20261020T042244_360547Z"
    r"\native_statistics_bootstrap_20261021T025138_331609Z"
)

OUTPUT_DIR = Path(r"C:\internnn\oracle_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCH_FILES = {
    "raw": NATIVE_DIR / "raw_native_stats.npz",
    "fixed_five_layer": NATIVE_DIR / "fixed_five_layer_native_stats.npz",
    "fixed_residual": NATIVE_DIR / "fixed_residual_native_stats.npz",
}

# Detections below this confidence are ignored when choosing a branch.
UTILITY_CONF_THRESHOLD = 0.25

# IoU threshold used for the oracle utility.
UTILITY_IOU_THRESHOLD = 0.50

# Metric recomputed after oracle selection.
EVAL_IOU_THRESHOLD = 0.50

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# NATIVE NPZ LOADING
# ----------------------------------------------------------------------

def get_iou_index(num_thresholds):
    """
    Return the index corresponding to EVAL_IOU_THRESHOLD.

    Ultralytics native statistics normally store ten IoU thresholds:
    0.50, 0.55, ..., 0.95.
    """
    index = int(round((EVAL_IOU_THRESHOLD - 0.50) / 0.05))

    if index < 0 or index >= num_thresholds:
        raise ValueError(
            f"Cannot select IoU {EVAL_IOU_THRESHOLD} from "
            f"{num_thresholds} stored IoU thresholds."
        )

    return index


def load_branch(path):
    """
    Load one native-statistics file using Ultralytics offset format.

    Expected keys:
      filenames       : [num_images]
      tp              : [num_predictions, num_iou_thresholds]
      conf            : [num_predictions]
      pred_cls        : [num_predictions]
      target_cls      : [num_targets]
      pred_offsets    : [num_images + 1]
      target_offsets  : [num_images + 1]
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing native statistics file: {path}")

    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    required = {
        "filenames",
        "tp",
        "conf",
        "pred_cls",
        "target_cls",
        "pred_offsets",
        "target_offsets",
    }

    missing = required - set(keys)
    if missing:
        raise KeyError(
            f"{path.name} is missing required keys: {sorted(missing)}. "
            f"Available keys: {keys}"
        )

    filenames = np.asarray(data["filenames"])
    tp_all = np.asarray(data["tp"])
    conf_all = np.asarray(data["conf"], dtype=float)
    pred_cls_all = np.asarray(data["pred_cls"])
    target_cls_all = np.asarray(data["target_cls"])
    pred_offsets = np.asarray(data["pred_offsets"], dtype=int)
    target_offsets = np.asarray(data["target_offsets"], dtype=int)

    num_images = len(filenames)

    if tp_all.ndim != 2:
        raise ValueError(
            f"Expected tp to have shape "
            f"[num_predictions, num_iou_thresholds], "
            f"but received shape {tp_all.shape}."
        )

    if len(pred_offsets) != num_images + 1:
        raise ValueError(
            f"Expected pred_offsets to have length {num_images + 1}, "
            f"but received length {len(pred_offsets)}."
        )

    if len(target_offsets) != num_images + 1:
        raise ValueError(
            f"Expected target_offsets to have length {num_images + 1}, "
            f"but received length {len(target_offsets)}."
        )

    iou_index = get_iou_index(tp_all.shape[1])
    tp_eval_all = tp_all[:, iou_index].astype(bool)

    print(f"\nLoaded: {path.name}")
    print(f"Available keys: {keys}")
    print(f"Images: {num_images}")
    print(f"Predictions: {len(conf_all)}")
    print(f"Targets: {len(target_cls_all)}")
    print(f"TP array shape: {tp_all.shape}")
    print(f"Using IoU index {iou_index} for {EVAL_IOU_THRESHOLD:.2f}")

    # Build per-image prediction records using cumulative offsets.
    records = defaultdict(list)

    for image_index in range(num_images):
        start = int(pred_offsets[image_index])
        end = int(pred_offsets[image_index + 1])

        for prediction_index in range(start, end):
            records[image_index].append(
                {
                    "confidence": float(conf_all[prediction_index]),
                    "predicted_class": int(pred_cls_all[prediction_index]),
                    "tp": bool(tp_eval_all[prediction_index]),
                }
            )

    # Build per-image target counts by class using cumulative offsets.
    target_counts = defaultdict(lambda: defaultdict(int))

    for image_index in range(num_images):
        start = int(target_offsets[image_index])
        end = int(target_offsets[image_index + 1])

        for target_index in range(start, end):
            class_id = int(target_cls_all[target_index])
            target_counts[image_index][class_id] += 1

    return {
        "filenames": filenames,
        "records": records,
        "target_counts": target_counts,
        "num_images": num_images,
    }

def utility_for_image(predictions):
    """
    Compute TP - FP at the configured confidence threshold.

    This rewards correctly matched detections and penalizes
    false positives.
    """
    tp = 0
    fp = 0

    for prediction in predictions:
        if prediction["confidence"] < UTILITY_CONF_THRESHOLD:
            continue

        if prediction["tp"]:
            tp += 1
        else:
            fp += 1

    return tp - fp, tp, fp


def compute_pooled_map50(image_to_branch, loaded_branches):
    """
    Compute pooled mAP50 after oracle branch selection.

    For each class:
      1. Collect selected detections across all images.
      2. Sort by descending confidence.
      3. Accumulate TP and FP using the native per-detection TP flag.
      4. Compute AP using standard precision-recall interpolation.

    This preserves the validator's original matching decisions.
    It does not rematch boxes.
    """
    selected_predictions = []

    for image_index, branch_name in image_to_branch.items():
        for prediction in loaded_branches[branch_name]["records"][
            image_index
        ]:
            selected_predictions.append(
                {
                    "image_index": image_index,
                    "branch": branch_name,
                    "confidence": prediction["confidence"],
                    "predicted_class": prediction["predicted_class"],
                    "tp": prediction["tp"],
                }
            )

    classes = sorted(
        set(
            prediction["predicted_class"]
            for prediction in selected_predictions
        )
    )

    ap_per_class = {}
    target_support_per_class = defaultdict(int)

    # Target support is taken from the selected branch for each image.
    for image_index, branch_name in image_to_branch.items():
        for class_id, count in loaded_branches[branch_name][
            "target_counts"
        ][image_index].items():
            target_support_per_class[class_id] += count

    for class_id in classes:
        class_predictions = [
            prediction
            for prediction in selected_predictions
            if prediction["predicted_class"] == class_id
        ]

        class_predictions.sort(
            key=lambda prediction: prediction["confidence"],
            reverse=True,
        )

        total_targets = target_support_per_class[class_id]

        if total_targets == 0:
            ap_per_class[class_id] = 0.0
            continue

        tp_cumulative = 0
        fp_cumulative = 0

        precisions = []
        recalls = []

        for prediction in class_predictions:
            if prediction["tp"]:
                tp_cumulative += 1
            else:
                fp_cumulative += 1

            precision = tp_cumulative / (tp_cumulative + fp_cumulative)
            recall = tp_cumulative / total_targets

            precisions.append(precision)
            recalls.append(recall)

        precisions = np.asarray(precisions)
        recalls = np.asarray(recalls)

        # Make precision monotonically non-increasing from high recall
        # to low recall.
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        ap = 0.0
        previous_recall = 0.0

        for precision, recall in zip(precisions, recalls):
            if recall > previous_recall:
                ap += (recall - previous_recall) * precision
                previous_recall = recall

        ap_per_class[class_id] = float(ap)

    map50 = (
        float(np.mean(list(ap_per_class.values())))
        if ap_per_class
        else 0.0
    )

    return (
        map50,
        ap_per_class,
        dict(target_support_per_class),
        selected_predictions,
    )


# ----------------------------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------------------------

def main():
    print("=" * 72)
    print("ORACLE BEST-OF-THREE ANALYSIS")
    print("=" * 72)
    print(f"Utility confidence threshold: {UTILITY_CONF_THRESHOLD}")
    print(f"Utility IoU threshold: {UTILITY_IOU_THRESHOLD}")
    print(f"Reported metric: mAP{EVAL_IOU_THRESHOLD:.2f}")
    print(f"Random seed: {RANDOM_SEED}")
    print(
        "NOTE: This oracle uses evaluation labels to select branches "
        "and is an optimistic diagnostic bound, not a deployable policy."
    )

    loaded_branches = {}
    branch_maps = {}
    branch_utilities = {}

    for branch_name, path in BRANCH_FILES.items():
        branch = load_branch(path)
        loaded_branches[branch_name] = branch

        # Native-style pooled mAP50 for this branch alone.
        identity_selection = {
            image_index: branch_name
            for image_index in range(branch["num_images"])
        }

        branch_map, _, _, _ = compute_pooled_map50(
            identity_selection,
            loaded_branches,
        )
        branch_maps[branch_name] = branch_map

        # Per-image utility.
        utilities = {}
        utility_totals = {"tp": 0, "fp": 0}

        for image_index in range(branch["num_images"]):
            utility, tp, fp = utility_for_image(
                branch["records"][image_index]
            )
            utilities[image_index] = utility
            utility_totals["tp"] += tp
            utility_totals["fp"] += fp

        branch_utilities[branch_name] = utilities

        print(
            f"{branch_name:20s} "
            f"native-style pooled mAP50={branch_map:.6f}, "
            f"utility TP={utility_totals['tp']}, "
            f"FP={utility_totals['fp']}"
        )

    # ------------------------------------------------------------------
    # Oracle selection
    # ------------------------------------------------------------------
    num_images = loaded_branches["raw"]["num_images"]

    for branch_name, branch in loaded_branches.items():
        if branch["num_images"] != num_images:
            raise ValueError(
                f"Branch {branch_name} has "
                f"{branch['num_images']} images, "
                f"but raw has {num_images}."
            )

    image_to_branch = {}
    selection_counts = defaultdict(int)
    selection_rows = []

    # Deterministic tie-breaking:
    # raw > fixed_residual > fixed_five_layer.
    priority = {
        "raw": 2,
        "fixed_residual": 1,
        "fixed_five_layer": 0,
    }

    for image_index in range(num_images):
        candidate_utilities = {
            branch_name: branch_utilities[branch_name][image_index]
            for branch_name in BRANCH_FILES
        }

        best_branch = max(
            candidate_utilities,
            key=lambda name: (
                candidate_utilities[name],
                priority[name],
            ),
        )

        image_to_branch[image_index] = best_branch
        selection_counts[best_branch] += 1

        selection_rows.append(
            {
                "image_index": image_index,
                "filename": str(
                    loaded_branches["raw"]["filenames"][image_index]
                ),
                "selected_branch": best_branch,
                "raw_utility": candidate_utilities["raw"],
                "fixed_five_layer_utility": candidate_utilities[
                    "fixed_five_layer"
                ],
                "fixed_residual_utility": candidate_utilities[
                    "fixed_residual"
                ],
            }
        )

    # ------------------------------------------------------------------
    # Oracle pooled metric
    # ------------------------------------------------------------------
    oracle_map50, oracle_ap, oracle_support, selected_predictions = (
        compute_pooled_map50(
            image_to_branch,
            loaded_branches,
        )
    )

    raw_map50 = branch_maps["raw"]
    oracle_minus_raw = oracle_map50 - raw_map50

    print("\n" + "=" * 72)
    print("ORACLE RESULTS")
    print("=" * 72)

    print("\nBranch selection counts:")
    for branch_name in BRANCH_FILES:
        print(
            f"  {branch_name:20s}: "
            f"{selection_counts[branch_name]} images"
        )

    print("\nNative-style pooled mAP50:")
    for branch_name, value in branch_maps.items():
        print(f"  {branch_name:20s}: {value:.6f}")

    print(f"\nOracle best-of-three mAP50: {oracle_map50:.6f}")
    print(f"Raw mAP50:                 {raw_map50:.6f}")
    print(f"Oracle minus raw:          {oracle_minus_raw:.6f}")

    print("\nOracle class-wise AP:")
    for class_id in sorted(oracle_ap):
        print(
            f"  class {class_id}: "
            f"AP={oracle_ap[class_id]:.6f}, "
            f"target support={oracle_support.get(class_id, 0)}"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    selection_csv = OUTPUT_DIR / "oracle_image_selection.csv"

    with selection_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "image_index,filename,selected_branch,"
            "raw_utility,fixed_five_layer_utility,"
            "fixed_residual_utility\n"
        )

        for row in selection_rows:
            f.write(
                f"{row['image_index']},"
                f"{row['filename']},"
                f"{row['selected_branch']},"
                f"{row['raw_utility']},"
                f"{row['fixed_five_layer_utility']},"
                f"{row['fixed_residual_utility']}\n"
            )

    summary = {
        "configuration": "oracle_best_of_three",
        "utility": "TP - FP",
        "utility_confidence_threshold": UTILITY_CONF_THRESHOLD,
        "utility_iou_threshold": UTILITY_IOU_THRESHOLD,
        "evaluated_iou_threshold": EVAL_IOU_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "num_images": num_images,
        "branch_native_style_map50": branch_maps,
        "oracle_map50": oracle_map50,
        "raw_map50": raw_map50,
        "oracle_minus_raw": oracle_minus_raw,
        "oracle_class_ap": oracle_ap,
        "oracle_target_support": oracle_support,
        "selection_counts": dict(selection_counts),
        "interpretation": (
            "Optimistic test-label-informed diagnostic bound. "
            "Not a deployable routing policy."
        ),
    }

    summary_json = OUTPUT_DIR / "oracle_summary.json"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVED] Image selection: {selection_csv}")
    print(f"[SAVED] Summary: {summary_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()