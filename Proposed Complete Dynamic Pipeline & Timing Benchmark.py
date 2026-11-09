import os
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# ==============================================================================
# 1. EXACT VERIFIED PATHS
# ==============================================================================
URPC_TEST_IMAGES    = Path(r"C:\internnn\URPC2020\images\test")
URPC_TEST_LABELS    = Path(r"C:\internnn\URPC2020\labels\test")

CLASSIFIER_MODEL    = Path(r"C:\internnn\efficientnet_classifier_fixed.keras")
YOLO_WEIGHTS        = Path(r"C:\Users\pppan\runs\detect\urpc2020_yolov8n-9\weights\best.pt")

WATERNET_WEIGHTS    = Path(r"checkpoints/waternet_best.pth")
UWCNN_WEIGHTS       = Path(r"checkpoints/uwcnn_best.pth")

WORK_DIR            = Path(r"C:\internnn\end_to_end_evaluation")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

PROCESSING_SIZE = (640, 640)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)

CATEGORIES = ["good", "low_light", "turbid"]

# ==============================================================================
# 2. RESTORATION MODELS
# ==============================================================================
class WaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

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
        return torch.clamp(x + alpha * out, 0.0, 1.0)

@torch.inference_mode()
def apply_pytorch_restoration(model, img, model_type="waternet"):
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, PROCESSING_SIZE)
    
    # Ensure correct color space normalization to prevent channel inversion 
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor_in = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    if model_type == "waternet":
        enhanced_t = model(tensor_in).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    elif model_type == "uwcnn":
        # Applies alpha 0.35 dampening for UWCNN residual over-saturation
        enhanced_t = model(tensor_in, alpha=0.35).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert RGB output tensor strictly back to BGR for accurate OpenCV saving
    bgr_out = cv2.cvtColor((enhanced_t * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr_out, (orig_w, orig_h))

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def create_yolo_yaml(target_folder: Path, yaml_file: Path):
    root_str = str(target_folder).replace('\\', '/')
    content = f"""
path: {root_str}
train: images
val: images
names:
  0: holothurian
  1: echinus
  2: starfish
  3: scallops
"""
    with open(yaml_file, "w") as f:
        f.write(content.strip())

def copy_labels(img_list, dest_dir: Path):
    lbl_out = dest_dir / "labels"
    lbl_out.mkdir(parents=True, exist_ok=True)
    matched = 0
    for p in img_list:
        lbl = URPC_TEST_LABELS / f"{p.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, lbl_out / f"{p.stem}.txt")
            matched += 1
    return matched

# ==============================================================================
# 4. MAIN PIPELINE EVALUATION
# ==============================================================================
def main():
    print("=" * 75)
    print("1. LOADING CLASSIFIER, DETECTOR, AND RESTORATION MODELS")
    print("=" * 75)

    clf = tf.keras.models.load_model(CLASSIFIER_MODEL, compile=False)
    yolo_model = YOLO(str(YOLO_WEIGHTS))

    waternet = WaterNet().to(DEVICE)
    waternet.load_state_dict(torch.load(WATERNET_WEIGHTS, map_location=DEVICE))
    waternet.eval()

    uwcnn = UWCNN().to(DEVICE)
    uwcnn.load_state_dict(torch.load(UWCNN_WEIGHTS, map_location=DEVICE))
    uwcnn.eval()

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_urpc_imgs = sorted([p for p in URPC_TEST_IMAGES.iterdir() if p.suffix.lower() in valid_exts])
    print(f"Total URPC2020 test images: {len(all_urpc_imgs)}")

    # Set up isolated directories for cleanly tracking each pipeline configuration
    dirs = {
        "raw": WORK_DIR / "1_raw",
        "waternet": WORK_DIR / "2_waternet",
        "uwcnn": WORK_DIR / "3_uwcnn",
        "pipeline": WORK_DIR / "4_pipeline"
    }

    for d in dirs.values():
        (d / "images").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("2. PROCESSING IMAGES ACROSS ALL CONFIGURATIONS")
    print("=" * 75)

    counts = {'good': 0, 'low_light': 0, 'turbid': 0}

    for img_path in tqdm(all_urpc_imgs, desc="Processing images"):
        # 1. Save Raw Baseline
        shutil.copy2(img_path, dirs["raw"] / "images" / img_path.name)

        img = cv2.imread(str(img_path))
        if img is None: continue

        # 2. Extract Classifier Prediction
        img_clf = load_img(img_path, target_size=(224, 224))
        arr = np.expand_dims(img_to_array(img_clf) / 255.0, axis=0)
        preds = clf.predict(arr, verbose=0)[0]
        cat_idx = int(np.argmax(preds))
        quality = CATEGORIES[min(cat_idx, 2)]
        counts[quality] += 1

        # 3. Generate Fixed WaterNet Image
        wn_img = apply_pytorch_restoration(waternet, img, model_type="waternet")
        cv2.imwrite(str(dirs["waternet"] / "images" / img_path.name), wn_img)

        # 4. Generate Fixed UWCNN Image
        uw_img = apply_pytorch_restoration(uwcnn, img, model_type="uwcnn")
        cv2.imwrite(str(dirs["uwcnn"] / "images" / img_path.name), uw_img)

        # 5. Route to Dynamic Pipeline
        if quality == 'low_light':
            cv2.imwrite(str(dirs["pipeline"] / "images" / img_path.name), wn_img)
        elif quality == 'turbid':
            cv2.imwrite(str(dirs["pipeline"] / "images" / img_path.name), uw_img)
        else: # 'good' -> Bypass
            cv2.imwrite(str(dirs["pipeline"] / "images" / img_path.name), img)

    print(f"\n[ROUTING SUMMARY] Good (Bypass): {counts['good']}, Low-Light: {counts['low_light']}, Turbid: {counts['turbid']}")

    print("\n" + "=" * 75)
    print("3. EVALUATING YOLOv8 DETECTION ACCURACY")
    print("=" * 75)

    summary = {}

    for name, split_dir in dirs.items():
        print(f"\n>>> EVALUATING CONFIGURATION: [{name.upper()}] <<<")
        matched = copy_labels(all_urpc_imgs, split_dir)
        yaml_file = WORK_DIR / f"{name}.yaml"
        create_yolo_yaml(split_dir, yaml_file)
        
        print(f"  Evaluating {len(all_urpc_imgs)} frames ({matched} labels matched)...")
        res = yolo_model.val(data=str(yaml_file), batch=16, imgsz=640, device="cpu", verbose=False)
        
        summary[name] = {
            "map50": res.box.map50,
            "map5095": res.box.map,
            "precision": res.box.mp,
            "recall": res.box.mr
        }

    print("\n" + "=" * 75)
    print("FINAL EXPERIMENTAL RESULTS: PIPELINE COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Pipeline Configuration':<25} | {'mAP50':<10} | {'mAP50-95':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 75)
    
    print(f"{'Baseline (Raw Images)':<25} | {summary['raw']['map50']:<10.4f} | {summary['raw']['map5095']:<10.4f} | {summary['raw']['precision']:<10.4f} | {summary['raw']['recall']:<10.4f}")
    print(f"{'Fixed WaterNet-All':<25} | {summary['waternet']['map50']:<10.4f} | {summary['waternet']['map5095']:<10.4f} | {summary['waternet']['precision']:<10.4f} | {summary['waternet']['recall']:<10.4f}")
    print(f"{'Fixed UWCNN-All':<25} | {summary['uwcnn']['map50']:<10.4f} | {summary['uwcnn']['map5095']:<10.4f} | {summary['uwcnn']['precision']:<10.4f} | {summary['uwcnn']['recall']:<10.4f}")
    print(f"{'Proposed Dynamic Pipeline':<25} | {summary['pipeline']['map50']:<10.4f} | {summary['pipeline']['map5095']:<10.4f} | {summary['pipeline']['precision']:<10.4f} | {summary['pipeline']['recall']:<10.4f}")
    print("=" * 75)

if __name__ == "__main__":
    main()