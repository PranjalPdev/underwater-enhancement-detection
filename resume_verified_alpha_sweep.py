import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import verified_alpha_sweep as experiment


ROOT = Path(r"C:\internnn")

CONFIGURATIONS = [
    "raw_alpha_0",
    "frozen_alpha_035",
    "generated_alpha_1",
]


def select_existing_run():
    candidates = []

    for folder in ROOT.glob("verified_alpha_sweep_*"):
        if not folder.is_dir():
            continue

        if not (folder / "artifact_manifest.csv").is_file():
            continue

        if not (folder / "metadata.json").is_file():
            continue

        if all(
            (folder / "datasets" / name / "images").is_dir()
            and (folder / "datasets" / name / "labels").is_dir()
            for name in CONFIGURATIONS
        ):
            candidates.append(folder)

    candidates.sort(key=lambda p: p.name)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "No completed generated dataset was found. "
            "Do not regenerate automatically; inspect the previous output."
        )

    print("Available generated runs:")
    for index, folder in enumerate(candidates, start=1):
        print(f"{index}: {folder}")

    value = input("Enter the run number to resume: ").strip()
    if not value.isdigit() or not 1 <= int(value) <= len(candidates):
        raise ValueError("Invalid run selection.")

    return candidates[int(value) - 1]


def verify_saved_artifacts(run):
    metadata = json.loads(
        (run / "metadata.json").read_text(encoding="utf-8")
    )

    for key, path in [
        ("residual_checkpoint_sha256", experiment.RESIDUAL_WEIGHTS),
        ("detector_checkpoint_sha256", experiment.DETECTOR_WEIGHTS),
    ]:
        if experiment.sha256(path) != metadata[key]:
            raise RuntimeError(f"Checkpoint changed: {path}")

    with open(
        run / "artifact_manifest.csv",
        newline="",
        encoding="utf-8",
    ) as f:
        records = list(csv.DictReader(f))

    if len(records) != experiment.EXPECTED_IMAGES:
        raise ValueError("Saved artifact manifest must contain 800 records.")

    names = [r["filename"].casefold() for r in records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate filenames in the artifact manifest.")

    hash_keys = {
        "raw_alpha_0": "raw_sha256",
        "frozen_alpha_035": "frozen_alpha035_sha256",
        "generated_alpha_1": "generated_alpha1_sha256",
    }

    for configuration in CONFIGURATIONS:
        dataset = run / "datasets" / configuration

        images = experiment.inventory(
            dataset / "images", experiment.IMAGE_EXTENSIONS
        )
        labels = experiment.inventory(dataset / "labels", {".txt"})

        if set(images) != set(names):
            raise RuntimeError(
                f"Image set differs from manifest: {configuration}"
            )

        expected_labels = {
            Path(r["filename"]).with_suffix(".txt").name.casefold()
            for r in records
        }
        if set(labels) != expected_labels:
            raise RuntimeError(
                f"Label set differs from manifest: {configuration}"
            )

        instances = 0

        for row in records:
            image_key = row["filename"].casefold()
            label_key = (
                Path(row["filename"])
                .with_suffix(".txt")
                .name.casefold()
            )

            if experiment.sha256(images[image_key]) != row[
                hash_keys[configuration]
            ]:
                raise RuntimeError(
                    f"Image hash mismatch: {images[image_key]}"
                )

            if experiment.sha256(labels[label_key]) != row["label_sha256"]:
                raise RuntimeError(
                    f"Label hash mismatch: {labels[label_key]}"
                )

            instances += len(
                experiment.read_labels(labels[label_key])
            )

        if instances != experiment.EXPECTED_INSTANCES:
            raise RuntimeError(
                f"{configuration}: expected 6579 instances, "
                f"found {instances}."
            )

        print(
            f"Verified {configuration}: "
            f"{len(images)} images, {instances} instances",
            flush=True,
        )


def main():
    run = select_existing_run()
    print(f"\nReusing saved datasets from:\n{run}", flush=True)

    verify_saved_artifacts(run)

    output = run / (
        "evaluation_resume_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)

    experiment.save_json(output / "resume_metadata.json", {
        "source_generated_run": str(run),
        "corrected_experiment_script_sha256": experiment.sha256(
            Path(experiment.__file__)
        ),
        "resume_script_sha256": experiment.sha256(Path(__file__)),
        "note": (
            "Existing images and labels verified against saved hashes. "
            "No image regeneration performed."
        ),
    })

    results = []

    for name in CONFIGURATIONS:
        print(f"\nEvaluating {name}", flush=True)

        result = experiment.evaluate(
            run / "datasets" / name,
            name,
            output,
        )
        results.append(result)

        experiment.save_json(
            output / "detector_metrics.json", results
        )
        experiment.write_csv(
            output / "detector_metrics.csv", results
        )

        print(json.dumps(result, indent=2), flush=True)

        if name in experiment.REFERENCE_MAP50:
            difference = abs(
                result["map50"] - experiment.REFERENCE_MAP50[name]
            )
            if difference > 2e-4:
                raise RuntimeError(
                    f"{name}: reference discrepancy "
                    f"{difference:.9f}. Results saved; stopped."
                )

    print("\nVERIFIED COEFFICIENT SWEEP")
    for result in results:
        print(
            f"alpha={result['alpha']:.2f}: "
            f"mAP50={result['map50']:.9f}, "
            f"mAP50:95={result['map50_95']:.9f}"
        )

    print(f"\nEvaluation reports saved to:\n{output}")


if __name__ == "__main__":
    main()