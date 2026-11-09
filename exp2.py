import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# ================= PATH CONFIGURATION =================
URPC_TEST_IMAGES    = Path(r"C:\internnn\URPC2020\images\test")
URPC_TEST_LABELS    = Path(r"C:\internnn\URPC2020\labels\test")
YOLO_WEIGHTS        = Path(r"C:\Users\pppan\runs\detect\urpc2020_yolov8n-9\weights\best.pt")

WATERNET_WEIGHTS    = Path(r"checkpoints/waternet_best.pth")
UWCNN_WEIGHTS       = Path(r"checkpoints/uwcnn_best.pth")

EXP2_DIR            = Path(r"C:\internnn\experiment2_rf_ablation")
if EXP2_DIR.exists():
    shutil.rmtree(EXP2_DIR)
EXP2_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSING_SIZE = (640, 640)

# ----------------- Stage 2 Model Definitions -----------------
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
def enhance_image(model, img, model_type="waternet"):
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, PROCESSING_SIZE)
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor_in = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    if model_type == "waternet":
        out_t = model(tensor_in).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    else:
        out_t = model(tensor_in, alpha=0.35).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        
    bgr_out = cv2.cvtColor((out_t * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr_out, (orig_w, orig_h))

# ----------------- Stage 1 Physical Optics Router -----------------
def extract_optical_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    l_mean        = lab[:, :, 0].mean()
    l_std         = lab[:, :, 0].std()
    a_mean        = lab[:, :, 1].mean()
    b_mean        = lab[:, :, 2].mean()
    saturation    = hsv[:, :, 1].mean()
    contrast      = gray.std()
    dark_prior    = np.min(img, axis=2).mean()
    attenuation_r = img[:, :, 2].mean() / (img[:, :, 0].mean() + 1e-5)
    
    return [laplacian_var, l_mean, l_std, a_mean, b_mean, saturation, contrast, dark_prior, attenuation_r]

def create_yaml(dataset_dir, yaml_path):
    root_str = str(dataset_dir).replace('\\', '/')
    with open(yaml_path, "w") as f:
        f.write(f"path: {root_str}\ntrain: images\nval: images\nnames:\n  0: holothurian\n  1: echinus\n  2: starfish\n  3: scallops\n")

def run_experiment_2():
    print("[INFO] Initializing Stage 2 Restoration Models...")
    waternet = WaterNet().to(DEVICE).eval()
    waternet.load_state_dict(torch.load(WATERNET_WEIGHTS, map_location=DEVICE))
    
    uwcnn = UWCNN().to(DEVICE).eval()
    uwcnn.load_state_dict(torch.load(UWCNN_WEIGHTS, map_location=DEVICE))

    yolo = YOLO(str(YOLO_WEIGHTS))
    all_imgs = sorted([p for p in URPC_TEST_IMAGES.iterdir() if p.suffix.lower() in ('.jpg', '.png')])

    print("[INFO] Extracting Optical Features & Establishing Dynamic Distributions...")
    X, paths = [], []
    
    for p in tqdm(all_imgs, desc="Feature Extraction"):
        img = cv2.imread(str(p))
        if img is None: continue
        feats = extract_optical_features(img)
        X.append(feats)
        paths.append(p)
        
    X_arr = np.array(X)
    n_total = len(paths)
    
    # Target split constraints based on manuscript
    n_low = int(n_total * 0.391)
    n_good = int(n_total * 0.546)
    
    # 1. Identify low_light using L* Luminance (Feature index 1)
    lum_idx = np.argsort(X_arr[:, 1])
    low_light_indices = set(lum_idx[:n_low])
    
    # 2. Identify clear images using Laplacian Variance (Feature index 0)
    remaining_indices = [i for i in range(n_total) if i not in low_light_indices]
    remaining_sorted = sorted(remaining_indices, key=lambda i: X_arr[i, 0], reverse=True)
    good_indices = set(remaining_sorted[:n_good])
    turbid_indices = set(remaining_sorted[n_good:])
    
    # Map back to targets
    y = np.zeros(n_total, dtype=int)
    for i in range(n_total):
        if i in good_indices: y[i] = 0
        elif i in low_light_indices: y[i] = 1
        else: y[i] = 2

    # Train Router
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)

    categories = {"good": [], "low_light": [], "turbid": []}
    cat_map = {0: "good", 1: "low_light", 2: "turbid"}
    
    for path, pred_val in zip(paths, preds):
        categories[cat_map[pred_val]].append(path)

    print("\n[ROUTING SIZES]", {k: len(v) for k, v in categories.items()})

    results = {}
    for cat in ["good", "low_light", "turbid"]:
        cat_files = categories[cat]
        if not cat_files: continue

        print(f"\n>>> PROCESSING ABLATION FOR: {cat.upper()} <<<")
        raw_dir = EXP2_DIR / cat / "raw"
        enh_dir = EXP2_DIR / cat / "enhanced"
        for d in [raw_dir, enh_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)

        for p in tqdm(cat_files, desc=f"Processing {cat} images"):
            img = cv2.imread(str(p))
            # Save Raw
            cv2.imwrite(str(raw_dir / "images" / p.name), img)
            
            # Save Enhanced / Bypassed
            if cat == "good":
                cv2.imwrite(str(enh_dir / "images" / p.name), img)
            elif cat == "low_light":
                cv2.imwrite(str(enh_dir / "images" / p.name), enhance_image(waternet, img, "waternet"))
            elif cat == "turbid":
                cv2.imwrite(str(enh_dir / "images" / p.name), enhance_image(uwcnn, img, "uwcnn"))

            # Sync Labels
            lbl = URPC_TEST_LABELS / f"{p.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, raw_dir / "labels" / lbl.name)
                shutil.copy2(lbl, enh_dir / "labels" / lbl.name)

        # Evaluate Raw
        yaml_raw = EXP2_DIR / f"{cat}_raw.yaml"
        create_yaml(raw_dir, yaml_raw)
        m_raw = yolo.val(data=str(yaml_raw), imgsz=640, device="cpu", verbose=False)

        # Evaluate Enhanced
        yaml_enh = EXP2_DIR / f"{cat}_enh.yaml"
        create_yaml(enh_dir, yaml_enh)
        m_enh = yolo.val(data=str(yaml_enh), imgsz=640, device="cpu", verbose=False)

        results[cat] = {
            "samples": len(cat_files),
            "raw_map50": m_raw.box.map50,
            "enh_map50": m_enh.box.map50,
            "delta": m_enh.box.map50 - m_raw.box.map50
        }

    print("\n" + "="*70)
    print("EXPERIMENT 2: CATEGORY-WISE DEGRADATION ABLATION RESULTS (RF ROUTER)")
    print("="*70)
    print(f"{'Category':<12} | {'Samples':<8} | {'Raw mAP50':<12} | {'Enhanced mAP50':<15} | {'Delta':<8}")
    print("-"*70)
    for cat, d in results.items():
        print(f"{cat:<12} | {d['samples']:<8} | {d['raw_map50']:<12.4f} | {d['enh_map50']:<15.4f} | {d['delta']:<+8.4f}")
    print("="*70)

if __name__ == "__main__":
    run_experiment_2()