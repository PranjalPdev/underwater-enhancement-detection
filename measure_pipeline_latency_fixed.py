# """Measure batch-1 CPU wall-clock latency for the underwater detection pipeline.

# This is a replacement for a truncated measure_pipeline_latency.py. It verifies
# saved routing decisions on the full 800-image manifest before timing, warms each
# configured path, records per-image stage timings, and writes an explicit final
# status to metadata.json.

# Review the paths and PREDICT_ARGS before running on the target machine.
# """

# # Set environment variables BEFORE importing TensorFlow, PyTorch, NumPy, OpenCV.
# import os

# CPU_THREADS = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
# os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
# os.environ["TF_NUM_INTRAOP_THREADS"] = str(CPU_THREADS)
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import ast
# import csv
# import hashlib
# import json
# import platform
# import random
# import shutil
# import tempfile
# import time
# import traceback
# from collections import Counter
# from datetime import datetime, timezone
# from pathlib import Path

# import cv2
# import numpy as np
# import tensorflow as tf
# import torch
# import torch.nn as nn
# import ultralytics
# from ultralytics import YOLO

# ROOT = Path(r"C:\internnn")
# RUN_DIR = ROOT / "temp_eval" / "proposed_pipeline_20261019T203221_524771Z"
# BENCHMARK_DIR = RUN_DIR / "detection_evaluation_20261020T042244_360547Z"
# BENCHMARK_SOURCE = ROOT / "evaluate_efficientnet_pipeline.py"
# CONFIG_PATH = RUN_DIR / "configuration.json"
# MANIFEST_PATH = RUN_DIR / "detection_manifest.json"
# ROUTING_PATH = RUN_DIR / "routing_manifest.csv"
# CLASS_MAPPING_PATH = RUN_DIR / "class_mapping.json"

# FIVE_WEIGHTS = ROOT / "checkpoints" / "waternet_best.pth"
# RESIDUAL_WEIGHTS = ROOT / "checkpoints" / "uwcnn_best.pth"
# DETECTOR_WEIGHTS = Path(
#     r"C:\Users\pppan\runs\detect\urpc2020_yolov8n-9\weights\best.pt"
# )
# NUM_IMAGES = 800
# REPEATS = 3
# WARMUP_CALLS = 5
# SEED = 42

# CONFIGURATIONS = ["raw", "resize_only", "fixed_five_layer", "fixed_residual", "dynamic"]
# CLASS_NAMES = ["holothurian", "echinus", "scallop", "starfish"]
# PREDICT_ARGS = {
#     "imgsz": 640,
#     "rect": False,
#     "batch": 1,
#     "device": "cpu",
#     "half": False,
#     "augment": False,
#     "conf": 0.001,
#     "iou": 0.7,
#     "max_det": 300,
#     "save": False,
#     "save_txt": False,
#     "save_conf": False,
#     "verbose": False,
#     "stream": False,
# }


# def sha256(path: Path) -> str:
#     h = hashlib.sha256()
#     with open(path, "rb") as f:
#         for block in iter(lambda: f.read(1024 * 1024), b""):
#             h.update(block)
#     return h.hexdigest()


# def sha256_text(value: str) -> str:
#     return hashlib.sha256(value.encode("utf-8")).hexdigest()


# def save_json(path: Path, value) -> None:
#     path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


# def save_csv(path: Path, rows) -> None:
#     if not rows:
#         path.write_text("", encoding="utf-8")
#         return
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(rows)


# def extract_processing_functions():
#     """Extract only the model/IO definitions from the existing benchmark script."""
#     if not BENCHMARK_SOURCE.is_file():
#         raise FileNotFoundError(BENCHMARK_SOURCE)
#     source = BENCHMARK_SOURCE.read_text(encoding="utf-8-sig")
#     tree = ast.parse(source, filename=str(BENCHMARK_SOURCE))
#     required = [
#         "FiveLayerEnhancementCNN",
#         "DenseResidualEnhancementCNN",
#         "load_enhancer",
#         "enhance",
#         "resize_only",
#         "read_image",
#         "write_image",
#     ]
#     nodes = []
#     for name in required:
#         found = [
#             node
#             for node in tree.body
#             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
#         ]
#         if len(found) != 1:
#             raise ValueError(
#                 f"Expected exactly one top-level definition of {name}; found {len(found)}"
#             )
#         nodes.append(found[0])
#     module = ast.Module(body=nodes, type_ignores=[])
#     ast.fix_missing_locations(module)
#     namespace = {
#         "__name__": "latency_benchmark_functions",
#         "os": os,
#         "Path": Path,
#         "shutil": shutil,
#         "torch": torch,
#         "nn": nn,
#         "np": np,
#         "cv2": cv2,
#         "DEVICE": torch.device("cpu"),
#         "ALPHA": 0.35,
#     }
#     exec(compile(module, str(BENCHMARK_SOURCE), "exec"), namespace)
#     return namespace


# def path_key(value) -> str:
#     return str(Path(value).resolve()).casefold()


# def load_records():
#     config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
#     manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
#     with open(ROUTING_PATH, newline="", encoding="utf-8-sig") as f:
#         route_rows = list(csv.DictReader(f))

#     if len(manifest) != 800:
#         raise ValueError(
#             f"Expected 800 detection-manifest records; got {len(manifest)}"
#         )
#     route_map = {path_key(row["source"]): row for row in route_rows}
#     if len(route_map) != len(route_rows):
#         raise ValueError("Duplicate sources in routing manifest.")
#     if {path_key(row["path"]) for row in manifest} != set(route_map):
#         raise ValueError("Detection/routing image identities differ.")
#     if config.get("routing_policy") != "argmax":
#         raise ValueError(
#             f"Expected routing_policy='argmax'; got {config.get('routing_policy')!r}"
#         )

#     for field, weights in [
#         ("five_layer_checkpoint_sha256", FIVE_WEIGHTS),
#         ("residual_checkpoint_sha256", RESIDUAL_WEIGHTS),
#     ]:
#         if not weights.is_file():
#             raise FileNotFoundError(weights)
#         if sha256(weights) != config[field]:
#             raise RuntimeError(f"Checkpoint hash mismatch: {weights}")
#     if float(config["residual_alpha"]) != 0.35:
#         raise ValueError("Unexpected residual coefficient; expected 0.35.")

#     records = []
#     for item in manifest:
#         source = Path(item["path"])
#         if not source.is_file():
#             raise FileNotFoundError(source)
#         if sha256(source) != item["file_sha256"]:
#             raise RuntimeError(f"Source image hash mismatch: {source}")
#         route_row = route_map[path_key(source)]
#         route = route_row["route"].strip().casefold()
#         records.append({"source": source, "saved_route": route})

#     valid_routes = {"good", "low_light", "turbid"}
#     unknown = sorted({r["saved_route"] for r in records} - valid_routes)
#     if unknown:
#         raise ValueError(f"Unexpected route labels in routing manifest: {unknown}")
#     if not 1 <= NUM_IMAGES <= len(records):
#         raise ValueError("NUM_IMAGES must be between 1 and 800.")
#     return config, records


# def summarize(timing_rows):
#     summaries = []
#     groups = [
#         (name, "all", [r for r in timing_rows if r["configuration"] == name])
#         for name in CONFIGURATIONS
#     ]
#     for route in ("good", "low_light", "turbid"):
#         subset = [
#             r
#             for r in timing_rows
#             if r["configuration"] == "dynamic" and r["route"] == route
#         ]
#         if subset:
#             groups.append(("dynamic", route, subset))

#     for configuration, route, rows in groups:
#         if not rows:
#             continue
#         totals = np.asarray([float(r["total_ms"]) for r in rows], dtype=float)
#         summaries.append(
#             {
#                 "configuration": configuration,
#                 "route_group": route,
#                 "observations": len(rows),
#                 "unique_images": len({r["filename"] for r in rows}),
#                 "mean_ms": float(np.mean(totals)),
#                 "median_ms": float(np.median(totals)),
#                 "p95_ms": float(np.percentile(totals, 95)),
#                 "std_ms": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
#                 "estimated_serial_service_rate_images_per_second": float(
#                     1000.0 / np.mean(totals)
#                 ),
#                 "mean_classifier_ms": float(
#                     np.mean([float(r["classifier_ms"]) for r in rows])
#                 ),
#                 "mean_processing_ms": float(
#                     np.mean([float(r["processing_ms"]) for r in rows])
#                 ),
#                 "mean_detector_call_ms": float(
#                     np.mean([float(r["detector_call_ms"]) for r in rows])
#                 ),
#             }
#         )
#     return summaries


# def main():
#     # Configure frameworks before loading models.
#     tf.config.set_visible_devices([], "GPU")
#     tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
#     tf.config.threading.set_inter_op_parallelism_threads(1)

#     torch.set_num_threads(CPU_THREADS)

#     interop_warning = None
#     try:
#         torch.set_num_interop_threads(1)
#     except RuntimeError as exc:
#         interop_warning = str(exc)
#         print(
#             "Warning: PyTorch inter-op threads could not be reset. "
#             "The effective value will be recorded.",
#             flush=True,
#         )

#     cv2.setNumThreads(1)

#     config, all_records = load_records()
#     processing = extract_processing_functions()

#     classifier_path = Path(config["model_output"])
#     if not classifier_path.is_file():
#         raise FileNotFoundError(classifier_path)

#     mapping = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8-sig"))
#     index_to_class = {int(index): name for name, index in mapping.items()}

#     if set(index_to_class) != {0, 1, 2}:
#         raise ValueError("Expected classifier output indices 0, 1, 2.")

#     if set(index_to_class.values()) != {"good", "low_light", "turbid"}:
#         raise ValueError("Unexpected classifier class mapping.")

#     run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
#     output = ROOT / f"pipeline_latency_final_{run_stamp}"
#     output.mkdir(parents=True, exist_ok=False)

#     metadata = {
#         "status": "initializing",
#         "started_at_utc": datetime.now(timezone.utc).isoformat(),
#         "sample_images": NUM_IMAGES,
#         "manifest_images": len(all_records),
#         "repetitions": REPEATS,
#         "seed": SEED,
#         "warmup_calls_per_configuration": WARMUP_CALLS,
#         "dynamic_warmup_calls_per_route": WARMUP_CALLS,
#         "requested_cpu_threads_per_framework": CPU_THREADS,
#         "requested_tensorflow_inter_op_threads": 1,
#         "requested_pytorch_inter_op_threads": 1,
#         "pytorch_inter_op_configuration_warning": interop_warning,
#         "python": platform.python_version(),
#         "tensorflow": tf.__version__,
#         "pytorch": torch.__version__,
#         "ultralytics": ultralytics.__version__,
#         "opencv": cv2.__version__,
#         "numpy": np.__version__,
#         "predict_arguments": PREDICT_ARGS,
#         "benchmark_source_sha256": sha256(BENCHMARK_SOURCE),
#         "classifier_sha256": sha256(classifier_path),
#         "five_layer_sha256": sha256(FIVE_WEIGHTS),
#         "residual_sha256": sha256(RESIDUAL_WEIGHTS),
#         "detector_sha256": sha256(DETECTOR_WEIGHTS),
#         "routing_manifest_sha256": sha256(ROUTING_PATH),
#         "detection_manifest_sha256": sha256(MANIFEST_PATH),
#         "configuration_sha256": sha256(CONFIG_PATH),
#         "class_mapping_sha256": sha256(CLASS_MAPPING_PATH),
#         "script_sha256": sha256(Path(__file__)),
#         "scope": (
#             "Batch-1 serial CPU wall-clock latency. Model loading, "
#             "route verification, warm-up, and report writing are "
#             "excluded. Timed runs include temporary image staging, "
#             "classification for dynamic routing, selected processing, "
#             "buffered file I/O, and the detector prediction API call. "
#             "The detector call includes input decoding, preprocessing, "
#             "inference, and postprocessing. Raw images are also copied "
#             "to the staging directory. All models remain resident."
#         ),
#     }
#     save_json(output / "metadata.json", metadata)

#     timing_rows = []

#     def thread_settings():
#         return {
#             "torch_intra_op": torch.get_num_threads(),
#             "torch_inter_op": torch.get_num_interop_threads(),
#             "tensorflow_intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
#             "tensorflow_inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
#             "opencv": cv2.getNumThreads(),
#         }

#     try:
#         print(
#             "Loading models; loading is excluded from timing...",
#             flush=True,
#         )

#         classifier = tf.keras.models.load_model(
#             str(classifier_path),
#             compile=False,
#         )

#         five = processing["load_enhancer"](
#             processing["FiveLayerEnhancementCNN"](),
#             FIVE_WEIGHTS,
#         )

#         residual = processing["load_enhancer"](
#             processing["DenseResidualEnhancementCNN"](),
#             RESIDUAL_WEIGHTS,
#         )

#         detector = YOLO(str(DETECTOR_WEIGHTS))

#         actual_names = [detector.names[i] for i in range(len(detector.names))]
#         if actual_names != CLASS_NAMES:
#             raise ValueError("Unexpected detector class mapping:\n" f"{actual_names}")

#         def classify(source):
#             image = tf.keras.utils.load_img(
#                 source,
#                 color_mode="rgb",
#                 target_size=(224, 224),
#                 interpolation="bilinear",
#             )

#             try:
#                 array = tf.keras.utils.img_to_array(image)
#             finally:
#                 image.close()

#             probabilities = classifier(
#                 array[None, ...],
#                 training=False,
#             ).numpy()[0]

#             return index_to_class[int(np.argmax(probabilities))]

#         print(
#             "Verifying classifier routes on all manifest images...",
#             flush=True,
#         )

#         route_audit = []

#         for index, record in enumerate(all_records, start=1):
#             predicted = classify(record["source"])

#             route_audit.append(
#                 {
#                     "source": str(record["source"]),
#                     "saved_route": record["saved_route"],
#                     "reloaded_classifier_route": predicted,
#                     "matches": predicted == record["saved_route"],
#                 }
#             )

#             if index % 100 == 0:
#                 print(
#                     f"Verified classifier routes: " f"{index}/{len(all_records)}",
#                     flush=True,
#                 )

#         save_csv(
#             output / "route_verification.csv",
#             route_audit,
#         )

#         mismatch_count = sum(not row["matches"] for row in route_audit)

#         metadata["route_verification_images"] = len(route_audit)
#         metadata["route_verification_mismatches"] = mismatch_count
#         metadata["route_verification_match_rate"] = 1.0 - mismatch_count / len(
#             route_audit
#         )

#         if mismatch_count:
#             metadata["status"] = "failed_route_verification"
#             metadata["error"] = (
#                 f"Reloaded classifier disagreed with saved routes "
#                 f"on {mismatch_count}/{len(route_audit)} images. "
#                 "No latency measurements were started."
#             )
#             save_json(output / "metadata.json", metadata)
#             raise RuntimeError(metadata["error"])

#         rng = random.Random(SEED)

#         if NUM_IMAGES == len(all_records):
#             records = list(all_records)
#         else:
#             records = rng.sample(all_records, NUM_IMAGES)

#         records.sort(
#             key=lambda record: (
#                 record["source"].name.casefold(),
#                 str(record["source"]).casefold(),
#             )
#         )

#         metadata["sample_images"] = len(records)
#         metadata["sample_route_counts"] = dict(
#             Counter(record["saved_route"] for record in records)
#         )

#         save_csv(
#             output / "sample_manifest.csv",
#             [
#                 {
#                     "source": str(record["source"]),
#                     "saved_route": record["saved_route"],
#                     "source_sha256": sha256(record["source"]),
#                 }
#                 for record in records
#             ],
#         )

#         metadata["status"] = "warming_up"
#         save_json(output / "metadata.json", metadata)

#         def process_image(source, destination, branch):
#             destination.parent.mkdir(
#                 parents=True,
#                 exist_ok=True,
#             )

#             if branch in ("raw", "good"):
#                 shutil.copy2(source, destination)
#                 return

#             image = processing["read_image"](source)

#             if branch == "resize_only":
#                 processed = processing["resize_only"](image)
#             elif branch == "low_light":
#                 processed = processing["enhance"](five, image)
#             elif branch == "turbid":
#                 processed = processing["enhance"](residual, image)
#             else:
#                 raise ValueError(f"Unknown processing branch: {branch}")

#             processing["write_image"](
#                 destination,
#                 processed,
#             )

#         def execute(
#             record,
#             configuration,
#             work_root,
#             repetition=0,
#             timed=True,
#         ):
#             source = record["source"]

#             unique_name = f"{sha256_text(str(source.resolve()))[:12]}_" f"{source.name}"
#             destination = work_root / configuration / unique_name

#             start_ns = time.perf_counter_ns()

#             classifier_ms = 0.0
#             route = "all"

#             if configuration == "dynamic":
#                 classifier_start = time.perf_counter_ns()

#                 route = classify(source)

#                 classifier_ms = (time.perf_counter_ns() - classifier_start) / 1e6

#                 branch = route

#             elif configuration == "raw":
#                 branch = "raw"

#             elif configuration == "resize_only":
#                 branch = "resize_only"

#             elif configuration == "fixed_five_layer":
#                 branch = "low_light"

#             elif configuration == "fixed_residual":
#                 branch = "turbid"

#             else:
#                 raise ValueError(f"Unknown configuration: {configuration}")

#             processing_start = time.perf_counter_ns()

#             process_image(
#                 source,
#                 destination,
#                 branch,
#             )

#             processing_ms = (time.perf_counter_ns() - processing_start) / 1e6

#             detector_start = time.perf_counter_ns()

#             detector.predict(
#                 source=str(destination),
#                 **PREDICT_ARGS,
#             )

#             detector_call_ms = (time.perf_counter_ns() - detector_start) / 1e6

#             total_ms = (time.perf_counter_ns() - start_ns) / 1e6

#             # These checks occur outside the measured interval.
#             if timed:
#                 if torch.get_num_threads() != CPU_THREADS:
#                     raise RuntimeError(
#                         "PyTorch intra-op thread count changed " "during the benchmark."
#                     )

#                 if configuration == "dynamic" and route != record["saved_route"]:
#                     raise RuntimeError(
#                         "Timed dynamic route differs from the "
#                         f"saved assignment: {source}"
#                     )

#             return {
#                 "configuration": configuration,
#                 "repetition": repetition,
#                 "filename": source.name,
#                 "source": str(source),
#                 "route": route,
#                 "saved_route": record["saved_route"],
#                 "classifier_ms": classifier_ms,
#                 "processing_ms": processing_ms,
#                 "detector_call_ms": detector_call_ms,
#                 "total_ms": total_ms,
#             }

#         with tempfile.TemporaryDirectory(
#             prefix="pipeline_latency_final_work_",
#             dir=str(ROOT),
#         ) as temporary_name:
#             work_root = Path(temporary_name)

#             print(
#                 "Warming models and processing paths...",
#                 flush=True,
#             )

#             first_record = records[0]

#             for configuration in (
#                 "raw",
#                 "resize_only",
#                 "fixed_five_layer",
#                 "fixed_residual",
#             ):
#                 for _ in range(WARMUP_CALLS):
#                     execute(
#                         first_record,
#                         configuration,
#                         work_root,
#                         timed=False,
#                     )

#             for route in ("good", "low_light", "turbid"):
#                 candidates = [
#                     record for record in all_records if record["saved_route"] == route
#                 ]

#                 if not candidates:
#                     raise ValueError(f"No image available to warm route {route}.")

#                 for _ in range(WARMUP_CALLS):
#                     execute(
#                         candidates[0],
#                         "dynamic",
#                         work_root,
#                         timed=False,
#                     )

#             # Set the intended count after predictor initialization.
#             torch.set_num_threads(CPU_THREADS)

#             metadata["effective_thread_settings_before_timing"] = thread_settings()

#             if torch.get_num_threads() != CPU_THREADS:
#                 raise RuntimeError(
#                     "Unexpected PyTorch intra-op thread count " "after warm-up."
#                 )

#             metadata["resolved_predictor_arguments"] = vars(
#                 detector.predictor.args
#             ).copy()

#             metadata["status"] = "running"
#             save_json(output / "metadata.json", metadata)

#             total_observations = len(records) * REPEATS * len(CONFIGURATIONS)

#             print(
#                 f"\nStarting timed benchmark:\n"
#                 f"  Images: {len(records)}\n"
#                 f"  Repetitions: {REPEATS}\n"
#                 f"  Configurations: {len(CONFIGURATIONS)}\n"
#                 f"  Expected observations: {total_observations}\n"
#                 f"  Effective thread settings: "
#                 f"{thread_settings()}\n",
#                 flush=True,
#             )

#             run_orders = []

#             for repetition in range(1, REPEATS + 1):
#                 run_rng = random.Random(SEED + repetition)

#                 image_order = list(records)
#                 configuration_order = list(CONFIGURATIONS)

#                 run_rng.shuffle(image_order)
#                 run_rng.shuffle(configuration_order)

#                 run_orders.append(
#                     {
#                         "repetition": repetition,
#                         "configuration_order": configuration_order,
#                         "image_order": [
#                             str(record["source"]) for record in image_order
#                         ],
#                     }
#                 )

#                 save_json(
#                     output / "measurement_orders.json",
#                     run_orders,
#                 )

#                 for configuration in configuration_order:
#                     for index, record in enumerate(
#                         image_order,
#                         start=1,
#                     ):
#                         timing_rows.append(
#                             execute(
#                                 record,
#                                 configuration,
#                                 work_root,
#                                 repetition=repetition,
#                                 timed=True,
#                             )
#                         )

#                         if index % 100 == 0:
#                             print(
#                                 f"Repeat {repetition}/{REPEATS} | "
#                                 f"{configuration} | "
#                                 f"{index}/{len(image_order)}",
#                                 flush=True,
#                             )

#                     save_csv(
#                         output / "timing_observations_partial.csv",
#                         timing_rows,
#                     )

#                     metadata["timed_observations_saved"] = len(timing_rows)
#                     save_json(
#                         output / "metadata.json",
#                         metadata,
#                     )

#                     print(
#                         f"Completed repetition {repetition}/{REPEATS}, "
#                         f"configuration {configuration}.",
#                         flush=True,
#                     )

#             metadata["effective_thread_settings_after_timing"] = thread_settings()

#         expected_count = len(records) * REPEATS * len(CONFIGURATIONS)

#         if len(timing_rows) != expected_count:
#             raise RuntimeError(
#                 f"Expected {expected_count} observations, " f"found {len(timing_rows)}."
#             )

#         save_csv(
#             output / "timing_observations.csv",
#             timing_rows,
#         )

#         summaries = summarize(timing_rows)
#         save_csv(output / "summary.csv", summaries)

#         repetition_summaries = []

#         for repetition in range(1, REPEATS + 1):
#             repetition_rows = [
#                 row for row in timing_rows if row["repetition"] == repetition
#             ]

#             for summary in summarize(repetition_rows):
#                 repetition_summaries.append(
#                     {
#                         "repetition": repetition,
#                         **summary,
#                     }
#                 )

#         save_csv(
#             output / "summary_by_repetition.csv",
#             repetition_summaries,
#         )

#         partial_path = output / "timing_observations_partial.csv"
#         if partial_path.exists():
#             partial_path.unlink()

#         metadata["status"] = "completed"
#         metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
#         metadata["timed_observations"] = len(timing_rows)

#         metadata["benchmark_label"] = (
#             "Batch-1, warm-model, staged-file CPU latency; "
#             "framework thread settings recorded"
#         )

#         metadata["summary_file"] = str(output / "summary.csv")
#         metadata["repetition_summary_file"] = str(output / "summary_by_repetition.csv")
#         metadata["timings_file"] = str(output / "timing_observations.csv")

#         save_json(output / "metadata.json", metadata)

#         print("\nFINAL LATENCY SUMMARY", flush=True)

#         for row in summaries:
#             if row["route_group"] != "all":
#                 continue

#             print(
#                 f"{row['configuration']:18s} "
#                 f"mean={row['mean_ms']:.3f} ms | "
#                 f"median={row['median_ms']:.3f} ms | "
#                 f"p95={row['p95_ms']:.3f} ms",
#                 flush=True,
#             )

#         print(
#             f"\nBenchmark completed.\n" f"Results saved to:\n{output}",
#             flush=True,
#         )

#     except Exception as exc:
#         if timing_rows:
#             try:
#                 save_csv(
#                     output / "timing_observations_partial.csv",
#                     timing_rows,
#                 )
#             except Exception:
#                 pass

#         if metadata.get("status") != "failed_route_verification":
#             metadata["status"] = "failed"
#             metadata["error"] = repr(exc)

#         metadata["traceback"] = traceback.format_exc()
#         metadata["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
#         metadata["timed_observations_saved"] = len(timing_rows)
#         metadata["effective_thread_settings_at_exit"] = thread_settings()

#         save_json(output / "metadata.json", metadata)

#         print(
#             f"\nBenchmark stopped. Diagnostic records are in:\n" f"{output}",
#             flush=True,
#         )
#         raise


# if __name__ == "__main__":
#     main()

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(r"C:\internnn\pipeline_latency_final_20261027T140507_636780Z")


def main():
    timings_path = RUN / "timing_observations.csv"
    metadata_path = RUN / "metadata.json"

    df = pd.read_csv(timings_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))

    required = {
        "configuration",
        "repetition",
        "source",
        "total_ms",
        "classifier_ms",
        "processing_ms",
        "detector_call_ms",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    timing_columns = [
        "total_ms",
        "classifier_ms",
        "processing_ms",
        "detector_call_ms",
    ]

    for column in timing_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all():
            raise ValueError(f"Nonfinite values in {column}")
        if (df[column] < 0).any():
            raise ValueError(f"Negative values in {column}")

    output = RUN / (
        "diagnostics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)

    df["stage_sum_ms"] = (
        df["classifier_ms"] + df["processing_ms"] + df["detector_call_ms"]
    )
    df["outside_stage_timers_ms"] = df["total_ms"] - df["stage_sum_ms"]

    summaries = []
    repetition_summaries = []
    top_rows = []

    for name, group in df.groupby("configuration", sort=False):
        ordered = group.sort_values("total_ms", ascending=False)
        values = group["total_ms"].to_numpy()
        total = float(values.sum())

        top_one_percent_n = max(1, int(np.ceil(len(group) * 0.01)))
        top_five_percent_n = max(1, int(np.ceil(len(group) * 0.05)))

        summaries.append(
            {
                "configuration": name,
                "observations": len(group),
                "unique_images": group["source"].nunique(),
                "mean_ms": float(values.mean()),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "maximum_ms": float(values.max()),
                "observations_over_10_seconds": int((values > 10000).sum()),
                "observations_over_60_seconds": int((values > 60000).sum()),
                "top_1_percent_share_of_total_time": (
                    float(ordered.head(top_one_percent_n)["total_ms"].sum()) / total
                    if total
                    else 0.0
                ),
                "top_5_percent_share_of_total_time": (
                    float(ordered.head(top_five_percent_n)["total_ms"].sum()) / total
                    if total
                    else 0.0
                ),
                "mean_classifier_ms": group["classifier_ms"].mean(),
                "mean_processing_ms": group["processing_ms"].mean(),
                "mean_detector_call_ms": group["detector_call_ms"].mean(),
                "maximum_outside_stage_timers_ms": (
                    group["outside_stage_timers_ms"].max()
                ),
                "minimum_outside_stage_timers_ms": (
                    group["outside_stage_timers_ms"].min()
                ),
            }
        )

        top = ordered.head(10).copy()
        stage_columns = [
            "classifier_ms",
            "processing_ms",
            "detector_call_ms",
        ]
        top["largest_recorded_stage"] = top[stage_columns].idxmax(axis=1)
        top_rows.append(top)

        for repetition, rep in group.groupby("repetition"):
            repetition_summaries.append(
                {
                    "configuration": name,
                    "repetition": repetition,
                    "observations": len(rep),
                    "unique_images": rep["source"].nunique(),
                    "mean_ms": rep["total_ms"].mean(),
                    "median_ms": rep["total_ms"].median(),
                    "p95_ms": rep["total_ms"].quantile(0.95),
                    "maximum_ms": rep["total_ms"].max(),
                    "over_60_seconds": int((rep["total_ms"] > 60000).sum()),
                }
            )

    summary = pd.DataFrame(summaries)
    repeats = pd.DataFrame(repetition_summaries)
    longest = pd.concat(top_rows, ignore_index=True)

    summary.to_csv(output / "distribution_summary.csv", index=False)
    repeats.to_csv(output / "repetition_summary.csv", index=False)
    longest.to_csv(output / "longest_observations.csv", index=False)

    df.nlargest(30, "total_ms").to_csv(output / "global_longest_30.csv", index=False)

    duplicate_count = int(
        df.duplicated(["configuration", "repetition", "source"]).sum()
    )

    checks = {
        "status": metadata.get("status"),
        "rows": len(df),
        "expected_rows_from_metadata": (
            metadata.get("sample_images", 0)
            * metadata.get("repetitions", 0)
            * df["configuration"].nunique()
        ),
        "duplicate_configuration_repetition_source_rows": duplicate_count,
        "effective_threads_before": metadata.get(
            "effective_thread_settings_before_timing"
        ),
        "effective_threads_after": metadata.get(
            "effective_thread_settings_after_timing"
        ),
        "materially_negative_stage_remainders": int(
            (df["outside_stage_timers_ms"] < -0.001).sum()
        ),
    }

    (output / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print("\nCHECKS")
    print(json.dumps(checks, indent=2))

    print("\nDISTRIBUTION SUMMARY")
    print(summary.round(3).to_string(index=False))

    print("\nBY REPETITION")
    print(repeats.round(3).to_string(index=False))

    print("\nLONGEST 10 OBSERVATIONS OVERALL")
    columns = [
        "configuration",
        "repetition",
        "source",
        "total_ms",
        "classifier_ms",
        "processing_ms",
        "detector_call_ms",
        "outside_stage_timers_ms",
    ]
    print(df.nlargest(10, "total_ms")[columns].round(3).to_string(index=False))

    print(f"\nDiagnostic reports saved to:\n{output}")


if __name__ == "__main__":
    main()
