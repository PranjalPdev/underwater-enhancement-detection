import csv
import hashlib
import inspect
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import ultralytics
from ultralytics.utils.metrics import ap_per_class


ROOT = Path(r"C:\internnn")

STATS_DIR = (
    ROOT
    / "temp_eval"
    / "proposed_pipeline_20261019T203221_524771Z"
    / "detection_evaluation_20261020T042244_360547Z"
    / "native_statistics_bootstrap_20261021T025138_331609Z"
)

FILES = {
    "raw": STATS_DIR / "raw_native_stats.npz",
    "resize": STATS_DIR / "resize_only_native_stats.npz",
    "residual": STATS_DIR / "fixed_residual_native_stats.npz",
    "five_layer": STATS_DIR / "fixed_five_layer_native_stats.npz",
}

EXPECTED_MAP50 = {
    "raw": 0.688634,
    "resize": 0.6882,
    "residual": 0.6445,
    "five_layer": 0.5142,
}

TOLERANCE = 2e-4
MAX_PASSES = 5
EPSILON = 0.0
EXPECTED_IMAGES = 800
EXPECTED_TARGETS = 6579

PREVIOUS_CONTROL_MAP50 = 0.7005963926024608


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_native(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    required = {
        "filenames",
        "tp",
        "conf",
        "pred_cls",
        "target_cls",
        "pred_offsets",
        "target_offsets",
    }

    with np.load(path, allow_pickle=False) as z:
        missing = required - set(z.files)
        if missing:
            raise ValueError(
                f"{path.name}: missing keys {sorted(missing)}"
            )
        try:
            d = {key: z[key] for key in required}
        except ValueError as exc:
            raise ValueError(
                f"{path.name}: an array requires pickle loading. "
                "Inspect the archive format rather than enabling "
                "pickle automatically."
            ) from exc

    if d["filenames"].ndim != 1:
        raise ValueError(f"{path.name}: filenames must be 1-D.")

    names = [
        (x.decode("utf-8") if isinstance(x, bytes) else str(x))
        .replace("\\", "/")
        for x in d["filenames"]
    ]

    if len(names) != EXPECTED_IMAGES:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_IMAGES} images, "
            f"found {len(names)}."
        )

    if len(names) != len(set(names)):
        raise ValueError(f"{path.name}: duplicate filenames.")

    tp = np.asarray(d["tp"])
    conf = np.asarray(d["conf"])
    pred = np.asarray(d["pred_cls"])
    target = np.asarray(d["target_cls"])

    if tp.ndim != 2 or tp.shape[1] != 10:
        raise ValueError(
            f"{path.name}: expected tp shape (N, 10), "
            f"found {tp.shape}."
        )

    if any(a.ndim != 1 for a in [conf, pred, target]):
        raise ValueError(
            f"{path.name}: conf/pred_cls/target_cls must be 1-D."
        )

    if not (len(tp) == len(conf) == len(pred)):
        raise ValueError(
            f"{path.name}: prediction array lengths differ."
        )

    if not np.all(np.isin(tp, [0, 1])):
        raise ValueError(f"{path.name}: tp contains nonbinary values.")

    if not all(np.isfinite(a).all() for a in [conf, pred, target]):
        raise ValueError(f"{path.name}: nonfinite statistics.")

    if np.any((conf < 0) | (conf > 1)):
        raise ValueError(f"{path.name}: confidence outside [0, 1].")

    for label, values in [("pred_cls", pred), ("target_cls", target)]:
        if np.any(values != np.floor(values)):
            raise ValueError(f"{path.name}: noninteger {label}.")
        if np.any((values < 0) | (values > 3)):
            raise ValueError(
                f"{path.name}: {label} outside expected IDs 0..3."
            )

    if len(target) != EXPECTED_TARGETS:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_TARGETS} targets, "
            f"found {len(target)}."
        )

    for key, total in [
        ("pred_offsets", len(conf)),
        ("target_offsets", len(target)),
    ]:
        offsets = d[key]
        valid = (
            offsets.ndim == 1
            and len(offsets) == len(names) + 1
            and np.issubdtype(offsets.dtype, np.integer)
            and offsets[0] == 0
            and offsets[-1] == total
            and np.all(np.diff(offsets) >= 0)
        )
        if not valid:
            raise ValueError(
                f"{path.name}: invalid cumulative {key}."
            )

    images = {}
    for i, name in enumerate(names):
        a, b = map(int, d["pred_offsets"][i:i + 2])
        c, e = map(int, d["target_offsets"][i:i + 2])

        images[name] = (
            tp[a:b].astype(bool, copy=False),
            conf[a:b],
            pred[a:b],
            target[c:e],
        )

    return names, images


def pooled_score(images):
    tp, conf, pred, target = [
        np.concatenate([image[k] for image in images], axis=0)
        for k in range(4)
    ]

    if len(target) == 0:
        raise ValueError("No ground-truth targets.")

    if len(conf) == 0:
        return {"map50": 0.0, "map50_95": 0.0}

    result = ap_per_class(
        tp,
        conf,
        pred,
        target,
        plot=False,
    )

    ap = np.asarray(result[5])
    if ap.ndim != 2 or ap.shape[1] != 10:
        raise RuntimeError(
            "Unexpected ap_per_class output schema."
        )

    return {
        "map50": float(ap[:, 0].mean()),
        "map50_95": float(ap.mean()),
    }


def save_selection(path, names, choices):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "selected_branch"])
        writer.writerows(zip(names, choices))


def run_search(label, branch_order, branches, names, output):
    folder = output / label
    folder.mkdir()

    choices = ["raw"] * len(names)
    selected = list(branches["raw"])
    baseline = pooled_score(selected)
    best = baseline["map50"]
    history = []
    converged = False

    print(f"\n{'=' * 70}")
    print(label.upper())
    print(f"Branches: {branch_order}")
    print(f"Starting raw mAP50: {best:.9f}")
    print(f"{'=' * 70}", flush=True)

    for pass_id in range(1, MAX_PASSES + 1):
        accepted = 0

        for i in range(len(names)):
            original_branch = choices[i]
            original_image = selected[i]
            local_branch = original_branch
            local_best = best

            # Test each alternative from the same current assignment.
            # Strict ties retain the incumbent, or the first improving
            # candidate in the predefined branch order.
            for candidate in branch_order:
                if candidate == original_branch:
                    continue

                selected[i] = branches[candidate][i]
                candidate_score = pooled_score(selected)["map50"]

                if candidate_score > local_best + EPSILON:
                    local_best = candidate_score
                    local_branch = candidate

            if local_branch != original_branch:
                selected[i] = branches[local_branch][i]
                choices[i] = local_branch
                best = local_best
                accepted += 1
            else:
                selected[i] = original_image

            if (i + 1) % 100 == 0:
                print(
                    f"Pass {pass_id}: {i + 1}/{len(names)}; "
                    f"best mAP50={best:.9f}",
                    flush=True,
                )

        current = pooled_score(selected)
        if abs(current["map50"] - best) > 1e-12:
            raise RuntimeError("Internal score-consistency check failed.")

        record = {
            "pass": pass_id,
            "accepted_image_switches": accepted,
            **current,
        }
        history.append(record)
        print(record, flush=True)

        save_json(folder / "history.json", history)
        save_selection(
            folder / "selection_latest.csv",
            names,
            choices,
        )

        if accepted == 0:
            converged = True
            break

    final = pooled_score(selected)
    summary = {
        "diagnostic": label,
        "branch_order": branch_order,
        "image_count": len(names),
        "raw_baseline": baseline,
        "selected": final,
        "gain_map50": final["map50"] - baseline["map50"],
        "gain_map50_95": (
            final["map50_95"] - baseline["map50_95"]
        ),
        "selected_counts": {
            branch: choices.count(branch)
            for branch in branch_order
        },
        "history": history,
        "converged": converged,
        "stopping_reason": (
            "complete_pass_without_improvement"
            if converged
            else "five_pass_limit"
        ),
        "max_passes": MAX_PASSES,
        "improvement_epsilon": EPSILON,
        "interpretation": (
            "In-sample label-informed diagnostic. "
            "Not a deployable router or global upper bound."
        ),
    }

    save_json(folder / "summary.json", summary)
    save_selection(folder / "selection.csv", names, choices)

    return summary


def main():
    output = ROOT / "matched_greedy_comparison" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    loaded = {}
    raw_names = None

    for branch, path in FILES.items():
        print(f"Loading {branch}: {path}", flush=True)
        names, images = load_native(path)
        loaded[branch] = (names, images)
        if branch == "raw":
            raw_names = names

    branches = {}

    for branch, (names, images) in loaded.items():
        if set(names) != set(raw_names):
            raise ValueError(
                f"{branch}: image identities differ from raw."
            )

        # All branches use the raw archive's stored image order.
        branches[branch] = [images[name] for name in raw_names]

        for name, raw_image, branch_image in zip(
            raw_names,
            [loaded["raw"][1][n] for n in raw_names],
            branches[branch],
        ):
            if not np.array_equal(
                np.sort(raw_image[3]),
                np.sort(branch_image[3]),
            ):
                raise ValueError(
                    f"{branch}: target classes differ for {name}."
                )

    baselines = {}
    print("\nBASELINE REPRODUCTION", flush=True)

    for branch, images in branches.items():
        value = pooled_score(images)
        baselines[branch] = value

        print(
            f"{branch:12s} "
            f"mAP50={value['map50']:.9f} "
            f"mAP50:95={value['map50_95']:.9f}",
            flush=True,
        )

        if abs(value["map50"] - EXPECTED_MAP50[branch]) > TOLERANCE:
            save_json(output / "baselines_failed.json", baselines)
            raise RuntimeError(
                f"{branch} baseline reproduction failed. "
                f"Observed {value['map50']:.9f}, "
                f"expected approximately {EXPECTED_MAP50[branch]:.9f}. "
                "Do not increase the tolerance to force a pass."
            )

    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ultralytics": ultralytics.__version__,
        "ap_per_class_sha256": hashlib.sha256(
            inspect.getsource(ap_per_class).encode()
        ).hexdigest(),
        "script_sha256": sha256(Path(__file__)),
        "input_files": {
            branch: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for branch, path in FILES.items()
        },
        "image_order": "raw NPZ stored filename order",
        "target_check": (
            "Per-image target-class multisets only; "
            "box coordinates are absent from these archives."
        ),
    }

    save_json(output / "metadata.json", metadata)
    save_json(output / "baselines.json", baselines)

    (output / "image_order.txt").write_text(
        "\n".join(raw_names) + "\n",
        encoding="utf-8",
    )

    control = run_search(
        "raw_resize_control",
        ["raw", "resize"],
        branches,
        raw_names,
        output,
    )

    reproduced_control = (
        abs(
            control["selected"]["map50"]
            - PREVIOUS_CONTROL_MAP50
        ) <= 1e-10
    )

    if not reproduced_control:
        save_json(
            output / "control_reproduction_warning.json",
            {
                "previous_control_map50": PREVIOUS_CONTROL_MAP50,
                "current_control_map50": (
                    control["selected"]["map50"]
                ),
                "message": (
                    "The previous control was not reproduced. "
                    "Stopped before the enhancement search."
                ),
            },
        )
        raise RuntimeError(
            "Raw/resize control differs from your completed run. "
            "Inspect inputs, metric implementation, and image order "
            "before continuing."
        )

    enhancement = run_search(
        "raw_residual_five_layer",
        ["raw", "residual", "five_layer"],
        branches,
        raw_names,
        output,
    )

    comparison = {
        "baselines": baselines,
        "control": control,
        "enhancement": enhancement,
        "previous_control_reproduced": reproduced_control,
        "comparison_note": (
            "Matched metric, image order, acceptance rule, and "
            "pass budget. Branch counts differ. Both results are "
            "in-sample; no corrected gain or significance claim "
            "is implied."
        ),
    }

    save_json(output / "comparison.json", comparison)

    with open(output / "comparison.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "diagnostic",
            "raw_map50",
            "selected_map50",
            "gain_map50",
            "selected_map50_95",
            "stopping_reason",
        ])

        for result in [control, enhancement]:
            writer.writerow([
                result["diagnostic"],
                result["raw_baseline"]["map50"],
                result["selected"]["map50"],
                result["gain_map50"],
                result["selected"]["map50_95"],
                result["stopping_reason"],
            ])

    print("\nFINAL MATCHED COMPARISON")
    for result in [control, enhancement]:
        print(
            f"{result['diagnostic']}: "
            f"mAP50={result['selected']['map50']:.9f}, "
            f"gain={result['gain_map50']:.9f}, "
            f"stop={result['stopping_reason']}",
            flush=True,
        )

    print(f"\nReports saved to: {output}")


if __name__ == "__main__":
    main()