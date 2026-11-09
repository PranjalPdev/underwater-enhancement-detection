import csv
import hashlib
import inspect
import json
import platform
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import ultralytics
from ultralytics.utils.metrics import ap_per_class


ROOT = Path(__file__).resolve().parent

EXPECTED_RAW_MAP50 = 0.688634
EXPECTED_RESIZE_MAP50 = 0.6882
REPRODUCTION_TOLERANCE = 2e-4

MAX_PASSES = 5
IMPROVEMENT_EPSILON = 0.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def choose_npz(title):
    stats_dir = (
        Path(r"C:\internnn")
        / "temp_eval"
        / "proposed_pipeline_20261019T203221_524771Z"
        / "detection_evaluation_20261020T042244_360547Z"
        / "native_statistics_bootstrap_20261021T025138_331609Z"
    )

    files = {
        "Select RAW detector native-statistics NPZ":
            stats_dir / "raw_native_stats.npz",

        "Select RESIZE-ONLY detector native-statistics NPZ":
            stats_dir / "resize_only_native_stats.npz",
    }

    if title not in files:
        raise ValueError(f"Unexpected file-selection request: {title}")

    selected = files[title]

    if not selected.is_file():
        raise FileNotFoundError(
            f"Required native-statistics file does not exist:\n{selected}"
        )

    print(f"\nAutomatically selected:\n{selected}", flush=True)
    return selected

def load_native(path):
    required = {
        "filenames", "tp", "conf", "pred_cls",
        "target_cls", "pred_offsets", "target_offsets"
    }

    with np.load(path, allow_pickle=False) as z:
        missing = required - set(z.files)
        if missing:
            raise ValueError(
                f"{path}\nMissing keys: {sorted(missing)}\n"
                f"Available keys: {z.files}"
            )

        try:
            d = {key: z[key] for key in required}
        except ValueError as exc:
            raise ValueError(
                f"{path}\nCannot load arrays without pickle. "
                "This script deliberately does not enable pickle. "
                "Check how this native archive was saved."
            ) from exc

    if d["filenames"].ndim != 1:
        raise ValueError("filenames must be one-dimensional.")

    names = [
        (x.decode("utf-8") if isinstance(x, bytes) else str(x))
        .replace("\\", "/")
        for x in d["filenames"]
    ]
    if not names or len(set(names)) != len(names):
        raise ValueError("Empty or duplicate image filenames.")

    tp = np.asarray(d["tp"])
    conf = np.asarray(d["conf"])
    pred = np.asarray(d["pred_cls"])
    target = np.asarray(d["target_cls"])

    if tp.ndim != 2 or tp.shape[1] != 10:
        raise ValueError(f"Expected tp shape (N,10), got {tp.shape}")
    if any(a.ndim != 1 for a in [conf, pred, target]):
        raise ValueError("conf, pred_cls, target_cls must be 1-D.")
    if not (len(tp) == len(conf) == len(pred)):
        raise ValueError("Prediction-array lengths differ.")
    if not np.all(np.isin(tp, [0, 1])):
        raise ValueError("tp must contain boolean/0/1 values.")
    if not all(np.isfinite(a).all() for a in [conf, pred, target]):
        raise ValueError("Non-finite statistics detected.")

    for key, length in [
        ("pred_offsets", len(conf)),
        ("target_offsets", len(target))
    ]:
        offsets = d[key]
        if (
            offsets.ndim != 1
            or len(offsets) != len(names) + 1
            or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0
            or offsets[-1] != length
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError(
                f"{key} must be a cumulative integer offset array "
                "with length number_of_images + 1."
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


def score(images):
    arrays = [
        np.concatenate([image[k] for image in images], axis=0)
        for k in range(4)
    ]
    tp, conf, pred, target = arrays

    if len(target) == 0:
        raise ValueError("No evaluation targets.")
    if len(conf) == 0:
        return 0.0, 0.0

    result = ap_per_class(tp, conf, pred, target, plot=False)
    ap = np.asarray(result[5])

    if ap.ndim != 2 or ap.shape[1] != 10:
        raise RuntimeError(
            "Unexpected ap_per_class return format. "
            "Use the original evaluation environment."
        )

    return float(ap[:, 0].mean()), float(ap.mean())


def main():
    gui = tk.Tk()
    gui.withdraw()

    try:
        print("Available NPZ files under the project:")
        for path in sorted(ROOT.rglob("*.npz")):
            print(path)

        raw_path = choose_npz(
            "Select RAW detector native-statistics NPZ"
        )
        if raw_path is None:
            print("Cancelled.")
            return

        resize_path = choose_npz(
            "Select RESIZE-ONLY detector native-statistics NPZ"
        )
        if resize_path is None:
            print("Cancelled.")
            return

        if raw_path.resolve() == resize_path.resolve():
            raise ValueError("Select two different native-statistics files.")

        print(f"\nRAW: {raw_path}")
        print(f"RESIZE: {resize_path}")

        names, raw_map = load_native(raw_path)
        resize_names, resize_map = load_native(resize_path)

        if set(names) != set(resize_names):
            raise ValueError("Raw and resize image identities differ.")
        if len(names) != 800:
            raise ValueError(
                f"Expected 800 evaluation images, found {len(names)}."
            )

        raw = [raw_map[name] for name in names]
        resize = [resize_map[name] for name in names]

        for name, a, b in zip(names, raw, resize):
            if not np.array_equal(np.sort(a[3]), np.sort(b[3])):
                raise ValueError(
                    f"Ground-truth class counts differ: {name}"
                )

        raw_score = score(raw)
        resize_score = score(resize)

        print(f"\nRaw reproduced mAP50:    {raw_score[0]:.9f}")
        print(f"Resize reproduced mAP50: {resize_score[0]:.9f}")

        for label, observed, expected in [
            ("raw", raw_score[0], EXPECTED_RAW_MAP50),
            ("resize", resize_score[0], EXPECTED_RESIZE_MAP50),
        ]:
            if abs(observed - expected) > REPRODUCTION_TOLERANCE:
                raise RuntimeError(
                    f"{label} baseline reproduction failed: "
                    f"observed={observed:.9f}, expected={expected:.9f}. "
                    "Do not change the tolerance merely to pass."
                )

        out = ROOT / "revision_greedy_control" / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        out.mkdir(parents=True, exist_ok=False)

        selected = raw.copy()
        choices = np.zeros(len(names), dtype=np.int8)
        best = raw_score[0]
        history = []

        print("\nBaseline gate passed. Starting greedy search.")
        print("Each candidate requires pooled AP recomputation.")

        for pass_id in range(1, MAX_PASSES + 1):
            accepted = 0

            for i in range(len(names)):
                candidate = 1 - int(choices[i])
                previous = selected[i]
                selected[i] = resize[i] if candidate else raw[i]

                candidate_score = score(selected)[0]

                if candidate_score > best + IMPROVEMENT_EPSILON:
                    choices[i] = candidate
                    best = candidate_score
                    accepted += 1
                else:
                    selected[i] = previous

                if (i + 1) % 100 == 0:
                    print(
                        f"Pass {pass_id}: {i + 1}/{len(names)} "
                        f"images; best mAP50={best:.9f}",
                        flush=True,
                    )

            record = {
                "pass": pass_id,
                "accepted_switches": accepted,
                "map50": best,
            }
            history.append(record)
            print(record, flush=True)

            if accepted == 0:
                break

        final = score(selected)

        summary = {
            "raw_file": str(raw_path),
            "resize_file": str(resize_path),
            "raw_sha256": sha256(raw_path),
            "resize_sha256": sha256(resize_path),
            "image_count": len(names),
            "raw_map50": raw_score[0],
            "raw_map50_95": raw_score[1],
            "resize_map50": resize_score[0],
            "resize_map50_95": resize_score[1],
            "selected_map50": final[0],
            "selected_map50_95": final[1],
            "gain_over_raw": final[0] - raw_score[0],
            "raw_selected": int((choices == 0).sum()),
            "resize_selected": int((choices == 1).sum()),
            "history": history,
            "image_order": "raw NPZ stored order",
            "max_passes": MAX_PASSES,
            "improvement_epsilon": IMPROVEMENT_EPSILON,
            "ultralytics_version": ultralytics.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "ap_per_class_source_sha256": hashlib.sha256(
                inspect.getsource(ap_per_class).encode()
            ).hexdigest(),
            "interpretation": (
                "In-sample label-informed selection diagnostic; "
                "not a deployable router or global upper bound."
            ),
        }

        (out / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        with open(out / "selection.csv", "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "selected_branch"])
            writer.writerows(
                (name, "resize" if choice else "raw")
                for name, choice in zip(names, choices)
            )

        print("\nFINAL SUMMARY")
        print(json.dumps(summary, indent=2))
        print(f"\nReports saved to: {out}")
    finally:
        gui.destroy()


if __name__ == "__main__":
    main()