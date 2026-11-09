"""
Residual-coefficient sweep.

alpha = 0.00 : raw images (identity)
alpha = 0.35 : existing fixed-residual images
alpha = 1.00 : newly generated full-residual images

All are evaluated with the same frozen raw-trained YOLOv8n detector.

Usage:
    python alpha_sweep.py
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ----------------------------------------------------------------------
# CONFIGURATION -- EDIT THESE PATHS
# ----------------------------------------------------------------------

RAW_IMAGE_DIR = Path(r"C:\internnn\URPC2020\images\test")
RAW_LABEL_DIR = Path(r"C:\internnn\URPC2020\labels\test")

# Directory where your existing fixed UWCNN images were generated
EXISTING_ALPHA_035_IMAGE_DIR = Path(r"C:\internnn\end_to_end_evaluation\3_uwcnn\images")

# The exact .pt file used for the frozen evaluation
DETECTOR_WEIGHTS = Path(r"C:\Users\pppan\runs\detect\urpc2020_yolov8n-9\weights\best.pt")

# The exact dense residual checkpoint
RESIDUAL_CHECKPOINT = Path(r"C:\internnn\checkpoints\uwcnn_best.pth")

OUTPUT_ROOT = Path(r"C:\internnn\alpha_sweep")

ALPHAS = [0.00, 0.35, 1.00]

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DETECTOR_ARGS = {
    "imgsz": 640,
    "batch": 16,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "rect": False,
    "verbose": False, # Set to false to reduce console spam
}

# ----------------------------------------------------------------------
# RESIDUAL MODEL DEFINITION
# ----------------------------------------------------------------------

class UWCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(80, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(96, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv2d(112, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(128, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv10 = nn.Conv2d(144, 3, 3, padding=1)

    def forward(self, x, alpha=0.35):
        x1 = self.conv1(x); x2 = self.conv2(x1); x3 = self.conv3(x2)
        cat1 = torch.cat([x1, x2, x3], dim=1); x4 = self.conv4(cat1)
        cat2 = torch.cat([cat1, x4], dim=1); x5 = self.conv5(cat2)
        cat3 = torch.cat([cat2, x5], dim=1); x6 = self.conv6(cat3)
        cat4 = torch.cat([cat3, x6], dim=1); x7 = self.conv7(cat4)
        cat5 = torch.cat([cat4, x7], dim=1); x8 = self.conv8(cat5)
        cat6 = torch.cat([cat5, x8], dim=1); x9 = self.conv9(cat6)
        cat7 = torch.cat([cat6, x9], dim=1); out = self.conv10(cat7)
        # Apply the residual coefficient directly inside the forward pass
        return torch.clamp(x + alpha * out, 0.0, 1.0)


def load_residual_model():
    """Load your dense residual enhancement checkpoint."""
    model = UWCNN().to(DEVICE)
    
    # Load state dict robustly
    state_dict = torch.load(RESIDUAL_CHECKPOINT, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded residual checkpoint: {RESIDUAL_CHECKPOINT}")
    return model

RESIDUAL_MODEL = None


def enhance_image_with_alpha(image_bgr, alpha):
    """
    Apply the dense residual enhancement branch with coefficient alpha.
    """
    global RESIDUAL_MODEL

    if RESIDUAL_MODEL is None:
        RESIDUAL_MODEL = load_residual_model()

    original_height, original_width = image_bgr.shape[:2]

    resized = cv2.resize(
        image_bgr,
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )

    # Convert to RGB normalized to [0, 1]
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.inference_mode():
        # Pass alpha directly to the model's forward function
        output = RESIDUAL_MODEL(tensor, alpha=alpha)

    # Revert tensor to image array
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).round().astype(np.uint8)

    # CRITICAL: Convert back from RGB to BGR to avoid YOLO color inversion
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    enhanced = cv2.resize(
        output_bgr,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    return enhanced


# ----------------------------------------------------------------------
# DATASET UTILITIES
# ----------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def copy_images(source_dir, target_image_dir):
    target_image_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in sorted(source_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        shutil.copy2(image_path, target_image_dir / image_path.name)
        copied += 1
    print(f"  copied {copied} images from {source_dir}")
    return copied


def copy_labels(target_label_dir):
    target_label_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for label_path in sorted(RAW_LABEL_DIR.iterdir()):
        if label_path.suffix.lower() != ".txt":
            continue
        shutil.copy2(label_path, target_label_dir / label_path.name)
        copied += 1
    print(f"  copied {copied} label files")
    return copied


def create_dataset_from_images(source_image_dir, dataset_root):
    image_dir = dataset_root / "images" / "val"
    label_dir = dataset_root / "labels" / "val"

    if image_dir.exists():
        shutil.rmtree(image_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)

    count = copy_images(source_image_dir, image_dir)
    copy_labels(label_dir)
    return count


def write_data_yaml(dataset_root):
    yaml_path = dataset_root / "data.yaml"
    # Ensure correct URPC target classes
    yaml_path.write_text(
        f"path: {str(dataset_root.resolve()).replace(chr(92), '/')}\n"
        "train: images/val\n"
        "val: images/val\n"
        "names:\n"
        "  0: holothurian\n"
        "  1: echinus\n"
        "  2: starfish\n"
        "  3: scallops\n",
        encoding="utf-8",
    )
    return yaml_path


# ----------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------

def evaluate_dataset(dataset_root, alpha):
    model = YOLO(str(DETECTOR_WEIGHTS))
    data_yaml = write_data_yaml(dataset_root)

    results = model.val(
        data=str(data_yaml),
        split="val",
        seed=RANDOM_SEED,
        **DETECTOR_ARGS,
    )

    metrics = {
        "alpha": alpha,
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    print(
        f"\nalpha={alpha:.2f}: "
        f"mAP50={metrics['map50']:.6f}, "
        f"mAP50-95={metrics['map50_95']:.6f}"
    )

    return metrics


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print("=" * 72)
    print("RESIDUAL-COEFFICIENT SWEEP")
    print("=" * 72)
    print(f"Alphas: {ALPHAS}")
    print(f"Device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")

    # Don't strictly fail on EXISTING_ALPHA_035_IMAGE_DIR if it's missing; 
    # we can recreate it if necessary, but we check critical ones.
    for required_path in [RAW_IMAGE_DIR, RAW_LABEL_DIR, DETECTOR_WEIGHTS]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []

    for alpha in ALPHAS:
        alpha_name = f"alpha_{alpha:.2f}".replace(".", "_")
        dataset_root = OUTPUT_ROOT / alpha_name

        print(f"\nPreparing alpha={alpha:.2f}")

        if alpha == 0.00:
            create_dataset_from_images(RAW_IMAGE_DIR, dataset_root)
        
        elif alpha == 0.35 and EXISTING_ALPHA_035_IMAGE_DIR.exists():
            # Reuse the exact images already evaluated as fixed_residual.
            create_dataset_from_images(EXISTING_ALPHA_035_IMAGE_DIR, dataset_root)

        else:
            image_dir = dataset_root / "images" / "val"
            label_dir = dataset_root / "labels" / "val"

            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            image_paths = sorted(
                path
                for path in RAW_IMAGE_DIR.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(f"  generating {len(image_paths)} enhanced images")

            for index, image_path in enumerate(image_paths, start=1):
                image_bgr = cv2.imread(str(image_path))

                if image_bgr is None:
                    raise RuntimeError(f"Could not read image: {image_path}")

                enhanced = enhance_image_with_alpha(image_bgr, alpha)

                cv2.imwrite(
                    str(image_dir / image_path.name),
                    enhanced,
                )

                if index % 100 == 0:
                    print(f"  processed {index}/{len(image_paths)}")

            copy_labels(label_dir)

        metrics = evaluate_dataset(dataset_root, alpha)
        all_results.append(metrics)

    summary_path = OUTPUT_ROOT / "alpha_sweep_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "configuration": "residual_coefficient_sweep",
                "alphas": ALPHAS,
                "random_seed": RANDOM_SEED,
                "detector_weights": str(DETECTOR_WEIGHTS),
                "results": all_results,
                "interpretation": (
                    "Tests whether the residual branch's relative "
                    "advantage is explained by damping toward the "
                    "identity operation."
                ),
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("ALPHA SWEEP SUMMARY")
    print("=" * 72)

    for result in all_results:
        print(
            f"alpha={result['alpha']:.2f}: "
            f"mAP50={result['map50']:.6f}, "
            f"mAP50-95={result['map50_95']:.6f}"
        )

    print(f"\n[SAVED] {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()