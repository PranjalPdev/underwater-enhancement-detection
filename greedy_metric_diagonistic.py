"""
Metric-aligned greedy routing diagnostic.

Starts with every image assigned to raw input. For each image, tests
switching it to fixed residual enhancement or fixed five-layer
enhancement. A switch is retained only if it increases pooled mAP50,
computed with the same native matching statistics used for the paper.

This is an in-sample, label-informed, greedy diagnostic.
It is NOT a deployable routing policy and NOT a guaranteed maximum.

Usage:
    python greedy_metric_diagnostic.py
"""

import json
import time
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

OUTPUT_DIR = Path(r"C:\internnn\greedy_metric_diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCH_FILES = {
    "raw": NATIVE_DIR / "raw_native_stats.npz",
    "fixed_five_layer": NATIVE_DIR / "fixed_five_layer_native_stats.npz",
    "fixed_residual": NATIVE_DIR / "fixed_residual_native_stats.npz",
}

EVAL_IOU_THRESHOLD = 0.50
MAX_PASSES = 5
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------

def get_iou_index(num_thresholds):
    index = int(round((EVAL_IOU_THRESHOLD - 0.50) / 0.05))

    if index < 0 or index >= num_thresholds:
        raise ValueError(
            f"Cannot select IoU {EVAL_IOU_THRESHOLD} from "
            f"{num_thresholds} stored thresholds."
        )

    return index


def load_branch(path):
    """
    Load one native-statistics file.

    Expected Ultralytics offset format:
      filenames: [num_images]
      tp: [num_predictions, num_iou_thresholds]
      conf: [num_predictions]
      pred_cls: [num_predictions]
      target_cls: [num_targets]
      pred_offsets: [num_images + 1]
      target_offsets: [num_images + 1]
    """
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
            f"{path.name} missing keys: {sorted(missing)}. "
            f"Available: {keys}"
        )

    filenames = np.asarray(data["filenames"])
    tp_all = np.asarray(data["tp"])
    conf_all = np.asarray(data["conf"], dtype=np.float32)
    pred_cls_all = np.asarray(data["pred_cls"], dtype=np.int64)
    target_cls_all = np.asarray(data["target_cls"], dtype=np.int64)
    pred_offsets = np.asarray(data["pred_offsets"], dtype=np.int64)
    target_offsets = np.asarray(data["target_offsets"], dtype=np.int64)

    num_images = len(filenames)

    if tp_all.ndim != 2:
        raise ValueError(
            f"Expected tp shape [predictions, thresholds], "
            f"received {tp_all.shape}."
        )

    if len(pred_offsets) != num_images + 1:
        raise ValueError(
            f"Expected pred_offsets length {num_images + 1}, "
            f"received {len(pred_offsets)}."
        )

    if len(target_offsets) != num_images + 1:
        raise ValueError(
            f"Expected target_offsets length {num_images + 1}, "
            f"received {len(target_offsets)}."
        )

    iou_index = get_iou_index(tp_all.shape[1])
    tp_eval_all = tp_all[:, iou_index].astype(np.int8)

    print(f"\nLoaded {path.name}")
    print(f"  images={num_images}, predictions={len(conf_all)}, "
          f"targets={len(target_cls_all)}")

    # Per-image prediction slices.
    image_predictions = []

    for image_index in range(num_images):
        start = int(pred_offsets[image_index])
        end = int(pred_offsets[image_index + 1])

        image_predictions.append(
            {
                "confidence": conf_all[start:end],
                "predicted_class": pred_cls_all[start:end],
                "tp": tp_eval_all[start:end],
            }
        )

    # Per-image target counts.
    image_target_counts = []

    for image_index in range(num_images):
        start = int(target_offsets[image_index])
        end = int(target_offsets[image_index + 1])

        counts = defaultdict(int)

        for class_id in target_cls_all[start:end]:
            counts[int(class_id)] += 1

        image_target_counts.append(dict(counts))

    return {
        "filenames": filenames,
        "image_predictions": image_predictions,
        "image_target_counts": image_target_counts,
        "num_images": num_images,
    }


# ----------------------------------------------------------------------
# POOLED mAP50 COMPUTATION
# ----------------------------------------------------------------------

def pooled_map50(image_to_branch, loaded):
    """
    Compute pooled mAP50 for a per-image branch assignment.

    Uses the native per-detection TP flag at EVAL_IOU_THRESHOLD.
    Detections from the selected branch for each image are combined,
    sorted by confidence, and evaluated as one pooled ranking.
    """
    confidences = []
    classes = []
    tp_flags = []

    for image_index, branch_name in image_to_branch.items():
        predictions = loaded[branch_name]["image_predictions"][image_index]

        confidences.append(predictions["confidence"])
        classes.append(predictions["predicted_class"])
        tp_flags.append(predictions["tp"])

    confidences = np.concatenate(confidences)
    classes = np.concatenate(classes)
    tp_flags = np.concatenate(tp_flags)

    # Target support comes from the selected branch for each image.
    target_support = defaultdict(int)

    for image_index, branch_name in image_to_branch.items():
        for class_id, count in loaded[branch_name][
            "image_target_counts"
        ][image_index].items():
            target_support[class_id] += count

    all_classes = sorted(
        set(classes.tolist()) | set(target_support.keys())
    )

    ap_per_class = {}

    for class_id in all_classes:
        total_targets = target_support[class_id]

        if total_targets == 0:
            ap_per_class[class_id] = 0.0
            continue

        mask = classes == class_id

        class_conf = confidences[mask]
        class_tp = tp_flags[mask]

        order = np.argsort(-class_conf, kind="stable")

        class_conf = class_conf[order]
        class_tp = class_tp[order]

        tp_cumulative = np.cumsum(class_tp)
        fp_cumulative = np.cumsum(1 - class_tp)

        precision = tp_cumulative / (tp_cumulative + fp_cumulative)
        recall = tp_cumulative / total_targets

        # Make precision monotonically non-increasing from high recall
        # to low recall.
        precision = np.maximum.accumulate(precision[::-1])[::-1]

        # Standard AP integration over increasing recall.
        recall_prev = np.concatenate(([0.0], recall[:-1]))
        ap = float(np.sum((recall - recall_prev) * precision))

        ap_per_class[class_id] = ap

    map50 = float(np.mean(list(ap_per_class.values())))

    return map50, ap_per_class, dict(target_support)


# ----------------------------------------------------------------------
# GREEDY SEARCH
# ----------------------------------------------------------------------

def main():
    print("=" * 72)
    print("METRIC-ALIGNED GREEDY ROUTING DIAGNOSTIC")
    print("=" * 72)
    print(f"Metric: pooled mAP{EVAL_IOU_THRESHOLD:.2f}")
    print(f"Maximum passes: {MAX_PASSES}")
    print(f"Random seed: {RANDOM_SEED}")
    print(
        "NOTE: in-sample, label-informed greedy diagnostic; "
        "not a deployable policy or guaranteed maximum."
    )

    loaded = {
        name: load_branch(path)
        for name, path in BRANCH_FILES.items()
    }

    num_images = loaded["raw"]["num_images"]

    for name, branch in loaded.items():
        if branch["num_images"] != num_images:
            raise ValueError(
                f"{name} has {branch['num_images']} images, "
                f"but raw has {num_images}."
            )

    # Start with all raw.
    assignment = np.full(num_images, "raw", dtype=object)

    current_map, current_ap, current_support = pooled_map50(
        {i: assignment[i] for i in range(num_images)},
        loaded,
    )

    print(f"\nInitial all-raw pooled mAP50: {current_map:.6f}")

    history = [
        {
            "pass": 0,
            "switches": 0,
            "map50": current_map,
            "delta_vs_initial": 0.0,
        }
    ]

    candidate_branches = ["fixed_residual", "fixed_five_layer"]

    for pass_index in range(1, MAX_PASSES + 1):
        print(f"\nPass {pass_index}")
        switches_this_pass = 0
        start_time = time.time()

        for image_index in range(num_images):
            original_branch = assignment[image_index]
            best_branch = original_branch
            best_map = current_map

            for candidate in candidate_branches:
                if candidate == original_branch:
                    continue

                assignment[image_index] = candidate

                trial_map, _, _ = pooled_map50(
                    {i: assignment[i] for i in range(num_images)},
                    loaded,
                )

                if trial_map > best_map + 1e-12:
                    best_map = trial_map
                    best_branch = candidate

            assignment[image_index] = best_branch

            if best_branch != original_branch:
                switches_this_pass += 1
                current_map = best_map

        print(
            f"  switches={switches_this_pass}, "
            f"mAP50={current_map:.6f}, "
            f"elapsed={time.time() - start_time:.1f}s"
        )

        history.append(
            {
                "pass": pass_index,
                "switches": switches_this_pass,
                "map50": current_map,
                "delta_vs_initial": current_map - history[0]["map50"],
            }
        )

        if switches_this_pass == 0:
            print("No improving single-image switch remained.")
            break

    final_map, final_ap, final_support = pooled_map50(
        {i: assignment[i] for i in range(num_images)},
        loaded,
    )

    selection_counts = defaultdict(int)

    for branch_name in assignment:
        selection_counts[branch_name] += 1

    print("\n" + "=" * 72)
    print("GREEDY DIAGNOSTIC RESULT")
    print("=" * 72)

    print("\nSelection counts:")
    for branch_name in BRANCH_FILES:
        print(f"  {branch_name:20s}: {selection_counts[branch_name]}")

    print(f"\nInitial all-raw mAP50: {history[0]['map50']:.6f}")
    print(f"Final greedy mAP50:    {final_map:.6f}")
    print(
        f"Greedy minus all-raw:  "
        f"{final_map - history[0]['map50']:.6f}"
    )

    print("\nGreedy class-wise AP:")
    for class_id in sorted(final_ap):
        print(
            f"  class {class_id}: AP={final_ap[class_id]:.6f}, "
            f"target support={final_support.get(class_id, 0)}"
        )

    # Save assignment.
    assignment_csv = OUTPUT_DIR / "greedy_assignment.csv"

    with assignment_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("image_index,filename,selected_branch\n")

        for image_index in range(num_images):
            f.write(
                f"{image_index},"
                f"{loaded['raw']['filenames'][image_index]},"
                f"{assignment[image_index]}\n"
            )

    summary = {
        "configuration": "greedy_metric_aligned_diagnostic",
        "metric": f"pooled_mAP_{EVAL_IOU_THRESHOLD:.2f}",
        "max_passes": MAX_PASSES,
        "random_seed": RANDOM_SEED,
        "initial_all_raw_map50": history[0]["map50"],
        "final_greedy_map50": final_map,
        "greedy_minus_all_raw": final_map - history[0]["map50"],
        "selection_counts": dict(selection_counts),
        "class_ap": final_ap,
        "target_support": final_support,
        "pass_history": history,
        "interpretation": (
            "In-sample, label-informed, greedy metric-aligned "
            "diagnostic. Not a deployable policy or guaranteed "
            "maximum."
        ),
    }

    summary_json = OUTPUT_DIR / "greedy_summary.json"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVED] Assignment: {assignment_csv}")
    print(f"[SAVED] Summary: {summary_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
    