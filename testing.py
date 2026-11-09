# # import time
# # import cv2
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # from pathlib import Path
# # from ultralytics import YOLO
# # from tensorflow.keras.models import load_model

# # CLASSIFIER_PATH  = Path(r"C:\internnn\efficientnet_classifier.keras")
# # YOLO_WEIGHTS      = Path(r"C:\Users\pppan\runs\detect\urpc2020_yolov8n-9\weights\best.pt")
# # WATERNET_WEIGHTS  = Path(r"checkpoints/waternet_best.pth")
# # UWCNN_WEIGHTS     = Path(r"checkpoints/uwcnn_best.pth")
# # TEST_IMAGES_DIR   = Path(r"C:\internnn\URPC2020\images\test")

# # DEVICE = torch.device("cpu") # Benchmarked directly on CPU

# # class WaterNet(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
# #             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
# #             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
# #             nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
# #             nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()
# #         )
# #     def forward(self, x): return self.net(x)

# # class UWCNN(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv4 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv5 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv6 = nn.Sequential(nn.Conv2d(80, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv7 = nn.Sequential(nn.Conv2d(96, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv8 = nn.Sequential(nn.Conv2d(112, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv9 = nn.Sequential(nn.Conv2d(128, 16, 3, padding=1), nn.ReLU(inplace=True))
# #         self.conv10 = nn.Conv2d(144, 3, 3, padding=1)

# #     def forward(self, x, alpha=0.35):
# #         x1 = self.conv1(x); x2 = self.conv2(x1); x3 = self.conv3(x2)
# #         cat1 = torch.cat([x1, x2, x3], dim=1); x4 = self.conv4(cat1)
# #         cat2 = torch.cat([cat1, x4], dim=1); x5 = self.conv5(cat2)
# #         cat3 = torch.cat([cat2, x5], dim=1); x6 = self.conv6(cat3)
# #         cat4 = torch.cat([cat3, x6], dim=1); x7 = self.conv7(cat4)
# #         cat5 = torch.cat([cat4, x7], dim=1); x8 = self.conv8(cat5)
# #         cat6 = torch.cat([cat5, x8], dim=1); x9 = self.conv9(cat6)
# #         cat7 = torch.cat([cat6, x9], dim=1); out = self.conv10(cat7)
# #         return torch.clamp(x + alpha * out, 0.0, 1.0)

# # def run_experiment_4():
# #     print("[EXP 4] Initializing CPU Latency Profiler...")
# #     clf = load_model(CLASSIFIER_PATH, compile=False)
# #     yolo = YOLO(str(YOLO_WEIGHTS))
# #     wn = WaterNet().to(DEVICE).eval()
# #     wn.load_state_dict(torch.load(WATERNET_WEIGHTS, map_location=DEVICE))
# #     uw = UWCNN().to(DEVICE).eval()
# #     uw.load_state_dict(torch.load(UWCNN_WEIGHTS, map_location=DEVICE))

# #     imgs = [cv2.imread(str(p)) for p in sorted(TEST_IMAGES_DIR.iterdir())[:100] if p.suffix.lower() in ('.jpg', '.png')]

# #     # Warmup
# #     dummy_t = torch.zeros((1, 3, 640, 640))
# #     with torch.no_grad():
# #         wn(dummy_t)
# #         uw(dummy_t)

# #     # 1. Stage 1 Latency
# #     t0 = time.perf_counter()
# #     for img in imgs:
# #         rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
# #         clf.predict(np.expand_dims(rgb, axis=0), verbose=0)
# #     t_stage1 = ((time.perf_counter() - t0) / len(imgs)) * 1000

# #     # 2. Stage 2 WaterNet Latency
# #     t0 = time.perf_counter()
# #     with torch.inference_mode():
# #         for img in imgs:
# #             t = torch.from_numpy(cv2.resize(img, (640, 640)).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
# #             wn(t)
# #     t_waternet = ((time.perf_counter() - t0) / len(imgs)) * 1000

# #     # 3. Stage 2 UWCNN Latency
# #     t0 = time.perf_counter()
# #     with torch.inference_mode():
# #         for img in imgs:
# #             t = torch.from_numpy(cv2.resize(img, (640, 640)).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
# #             uw(t)
# #     t_uwcnn = ((time.perf_counter() - t0) / len(imgs)) * 1000

# #     # 4. Stage 3 YOLOv8 Latency
# #     t0 = time.perf_counter()
# #     for img in imgs:
# #         yolo(img, verbose=False, imgsz=640)
# #     t_yolo = ((time.perf_counter() - t0) / len(imgs)) * 1000

# #     # Pipeline Totals (assuming 54.6% bypass, 39.1% WaterNet, 6.3% UWCNN)
# #     t_fixed_wn = t_waternet + t_yolo
# #     t_fixed_uw = t_uwcnn + t_yolo
# #     t_dynamic  = t_stage1 + (0.391 * t_waternet) + (0.063 * t_uwcnn) + t_yolo

# #     print("\n" + "="*60)
# #     print("EXPERIMENT 4: LATENCY AND THROUGHPUT PROFILE (CPU)")
# #     print("="*60)
# #     print(f"Stage 1 Classifier Latency:        {t_stage1:.2f} ms")
# #     print(f"Stage 2 WaterNet Latency:          {t_waternet:.2f} ms")
# #     print(f"Stage 2 UWCNN Latency:             {t_uwcnn:.2f} ms")
# #     print(f"Stage 3 YOLOv8 Latency:            {t_yolo:.2f} ms")
# #     print("-"*60)
# #     print(f"Fixed WaterNet Pipeline:           {t_fixed_wn:.2f} ms | {1000/t_fixed_wn:.2f} FPS")
# #     print(f"Fixed UWCNN Pipeline:              {t_fixed_uw:.2f} ms | {1000/t_fixed_uw:.2f} FPS")
# #     print(f"Proposed Dynamic Pipeline:         {t_dynamic:.2f} ms | {1000/t_dynamic:.2f} FPS")
# #     print(f"Speedup vs. Fixed UWCNN:           +{((t_fixed_uw - t_dynamic) / t_dynamic) * 100:.1f}%")
# #     print("="*60)

# # if __name__ == "__main__":
# # # #     run_experiment_4()
# import hashlib
# import json
# from datetime import datetime, timezone
# from pathlib import Path

# import numpy as np


# RUN_DIR = Path(
#     r"C:\internnn\temp_eval"
#     r"\proposed_pipeline_20261019T203221_524771Z"
# )

# SOURCE_LABELS = Path(r"C:\internnn\URPC2020\labels\test")

# CLEAN_LABELS = RUN_DIR / "labels_cleaned_zero_height_v1"
# AUDIT_PATH = RUN_DIR / "label_cleanup_zero_height_v1.json"

# MANIFEST_PATH = RUN_DIR / "detection_manifest.json"

# EXPECTED_EXCLUSIONS = {
#     ("000040.txt", 21):
#         "2 0.894775390625 0.75390625 0.00048828125 0.0",

#     ("000742.txt", 16):
#         "1 0.34791666666666665 0.7087962962962964 "
#         "0.0005208333333333333 0.0",
# }


# def sha256_bytes(data):
#     return hashlib.sha256(data).hexdigest()


# def validate_row(line, path, line_number):
#     fields = line.split()

#     if len(fields) != 5:
#         raise ValueError(
#             f"{path.name}:{line_number}: expected five fields"
#         )

#     values = np.asarray(fields, dtype=float)

#     if not np.isfinite(values).all():
#         raise ValueError(
#             f"{path.name}:{line_number}: NaN or infinity"
#         )

#     class_id, x, y, width, height = values

#     if class_id != int(class_id) or not 0 <= class_id < 4:
#         raise ValueError(
#             f"{path.name}:{line_number}: invalid class ID"
#         )

#     if not (
#         0 <= x <= 1
#         and 0 <= y <= 1
#         and 0 < width <= 1
#         and 0 < height <= 1
#     ):
#         raise ValueError(
#             f"{path.name}:{line_number}: invalid box: {line}"
#         )


# def main():
#     if CLEAN_LABELS.exists() or AUDIT_PATH.exists():
#         raise FileExistsError(
#             "Cleaned output or audit already exists. "
#             "Inspect it rather than overwriting it."
#         )

#     manifest = json.loads(
#         MANIFEST_PATH.read_text(encoding="utf-8")
#     )

#     prepared = []
#     excluded = []
#     encountered = set()
#     seen_names = set()
#     annotation_count = 0

#     # Validate everything before writing output files.
#     for record in manifest:
#         image_path = Path(record["path"])
#         label_name = f"{image_path.stem}.txt"

#         if label_name in seen_names:
#             raise ValueError(
#                 f"Duplicate label filename in manifest: {label_name}"
#             )
#         seen_names.add(label_name)

#         source = SOURCE_LABELS / label_name

#         if not source.is_file():
#             raise FileNotFoundError(source)

#         original_bytes = source.read_bytes()
#         lines = original_bytes.decode("utf-8-sig").splitlines()

#         retained_lines = []
#         changed = False

#         for number, line in enumerate(lines, start=1):
#             key = (label_name, number)

#             if key in EXPECTED_EXCLUSIONS:
#                 expected = EXPECTED_EXCLUSIONS[key]

#                 if line.strip() != expected:
#                     raise ValueError(
#                         f"Expected row has changed: {label_name}:{number}\n"
#                         f"Expected: {expected!r}\n"
#                         f"Found:    {line!r}"
#                     )

#                 excluded.append({
#                     "file": label_name,
#                     "original_line": number,
#                     "annotation": line,
#                     "reason": "Zero-height bounding box",
#                 })
#                 encountered.add(key)
#                 changed = True
#                 continue

#             if line.strip():
#                 validate_row(line, source, number)
#                 annotation_count += 1

#             retained_lines.append(line)

#         if changed:
#             cleaned_text = "\n".join(retained_lines)
#             if retained_lines:
#                 cleaned_text += "\n"
#             output_bytes = cleaned_text.encode("utf-8")
#         else:
#             # Preserve every unaffected label file byte-for-byte.
#             output_bytes = original_bytes

#         prepared.append({
#             "name": label_name,
#             "output_bytes": output_bytes,
#             "source_sha256": sha256_bytes(original_bytes),
#             "cleaned_sha256": sha256_bytes(output_bytes),
#             "changed": changed,
#         })

#     if encountered != set(EXPECTED_EXCLUSIONS):
#         missing = set(EXPECTED_EXCLUSIONS) - encountered
#         raise RuntimeError(
#             f"Expected exclusion rows were not found: {missing}"
#         )

#     CLEAN_LABELS.mkdir(parents=True, exist_ok=False)

#     for item in prepared:
#         (CLEAN_LABELS / item["name"]).write_bytes(
#             item["output_bytes"]
#         )

#     audit = {
#         "created_utc": datetime.now(timezone.utc).isoformat(),
#         "source_labels": str(SOURCE_LABELS),
#         "cleaned_labels": str(CLEAN_LABELS),
#         "manifest": str(MANIFEST_PATH),
#         "images_retained": len(manifest),
#         "label_files_written": len(prepared),
#         "valid_annotations_retained": annotation_count,
#         "annotations_excluded": len(excluded),
#         "policy": (
#             "Exclude only the two explicitly identified zero-height "
#             "rows. Retain all images and other annotations. "
#             "Original label files remain unchanged."
#         ),
#         "excluded_rows": excluded,
#         "file_hashes": [
#             {
#                 key: value
#                 for key, value in item.items()
#                 if key != "output_bytes"
#             }
#             for item in prepared
#         ],
#     }

#     AUDIT_PATH.write_text(
#         json.dumps(audit, indent=2),
#         encoding="utf-8",
#     )

#     print("Images retained:", len(manifest))
#     print("Label files written:", len(prepared))
#     print("Annotations excluded:", len(excluded))
#     print("Valid annotations retained:", annotation_count)
#     print("Cleaned labels:", CLEAN_LABELS)
#     print("Audit:", AUDIT_PATH)


# if __name__ == "__main__":
# #     main()
# from pathlib import Path

# import yaml
# from ultralytics import YOLO

# weights = Path(
#     r"C:\Users\pppan\runs\detect"
#     r"\urpc2020_yolov8n-9\weights\best.pt"
# )

# model = YOLO(str(weights))
# print("CHECKPOINT CLASS MAPPING:")
# print(model.names)

# args_path = weights.parent.parent / "args.yaml"
# print("\nTRAINING ARGUMENT FILE:", args_path)

# if args_path.exists():
#     args = yaml.safe_load(
#         args_path.read_text(encoding="utf-8")
#     )
#     print("Training data YAML:", args.get("data"))

#     data_value = args.get("data")
#     if isinstance(data_value, str):
#         data_path = Path(data_value)

#         if data_path.is_file():
#             dataset = yaml.safe_load(
#                 data_path.read_text(encoding="utf-8")
#             )
#             print("\nCURRENT CONTENTS OF TRAINING DATA YAML:")
#             print(yaml.safe_dump(dataset, sort_keys=False))
#         else:
#             print(
#                 "\nThe recorded YAML path is unavailable from "
#                 "this working directory. Locate it manually."
#             )
# else:
#     print("args.yaml was not found.")
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ultralytics

from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou


EVAL_DIR = Path(
    r"C:\internnn\temp_eval"
    r"\proposed_pipeline_20261019T203221_524771Z"
    r"\detection_evaluation_20261020T042244_360547Z"
)

LABEL_DIR = EVAL_DIR / "datasets" / "raw" / "labels"

stamp = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%S_%fZ"
)
OUTPUT = EVAL_DIR / f"metric_diagnostic_{stamp}.txt"


def get_source(owner, name):
    method = getattr(owner, name, None)
    if method is None:
        return f"{owner.__name__}.{name}: not present\n"

    try:
        source = inspect.getsource(method)
    except (OSError, TypeError) as error:
        source = f"Source unavailable: {error}"

    return (
        f"\n{'=' * 72}\n"
        f"{owner.__name__}.{name}\n"
        f"{'=' * 72}\n"
        f"{source}\n"
    )


sections = [
    f"Ultralytics version: {ultralytics.__version__}\n"
]

# Include all relevant preparation/export methods that exist.
for method_name in [
    "_prepare_batch",
    "_prepare_pred",
    "_process_batch",
    "update_metrics",
    "pred_to_json",
    "save_one_json",
    "get_stats",
]:
    sections.append(
        get_source(DetectionValidator, method_name)
    )

sections.append(
    get_source(BaseValidator, "match_predictions")
)

sections.append(
    "\nBOX_IOU IMPLEMENTATION\n"
    + inspect.getsource(box_iou)
)

# Record where IoU thresholds are initialized.
for owner in [BaseValidator, DetectionValidator]:
    source = inspect.getsource(owner)
    relevant = [
        line
        for line in source.splitlines()
        if "iouv" in line or "linspace" in line
    ]
    sections.append(
        f"\n{owner.__name__}: IoU initialization references\n"
        + "\n".join(relevant)
        + "\n"
    )

outside_boxes = []
duplicate_rows = []
total_boxes = 0

for path in sorted(LABEL_DIR.glob("*.txt")):
    seen = set()

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        values = tuple(float(v) for v in line.split())

        if len(values) != 5:
            raise ValueError(
                f"Unexpected annotation: {path}:{line_number}"
            )

        if values in seen:
            duplicate_rows.append({
                "file": path.name,
                "line": line_number,
            })
        seen.add(values)

        class_id, x, y, width, height = values
        total_boxes += 1

        corners = [
            x - width / 2,
            y - height / 2,
            x + width / 2,
            y + height / 2,
        ]

        # Tiny tolerance avoids flagging numerical noise alone.
        if (
            corners[0] < -1e-6
            or corners[1] < -1e-6
            or corners[2] > 1 + 1e-6
            or corners[3] > 1 + 1e-6
        ):
            outside_boxes.append({
                "file": path.name,
                "line": line_number,
                "annotation": line,
                "normalized_xyxy": corners,
            })

summary = {
    "label_files": len(list(LABEL_DIR.glob("*.txt"))),
    "annotation_rows": total_boxes,
    "boxes_extending_outside_image": len(outside_boxes),
    "duplicate_annotation_rows": len(duplicate_rows),
    "outside_box_examples": outside_boxes[:10],
}

sections.append(
    "\nLABEL AUDIT\n"
    + json.dumps(summary, indent=2)
    + "\n"
)

OUTPUT.write_text(
    "\n".join(sections),
    encoding="utf-8",
)

print(json.dumps(summary, indent=2))
print("\nSaved installed-source diagnostic:")
print(OUTPUT)