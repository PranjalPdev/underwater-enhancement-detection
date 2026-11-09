# step2_evaluate_benchmark.py
import os
import shutil
import time
import json
import joblib
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from common_feature import extract_optical_features_from_bgr

# ================= USER CONFIGURATION =================
DETECTION_IMG_DIR   = Path(r"C:\internnn\URPC2020\images\test")
DETECTION_LBL_DIR   = Path(r"C:\internnn\URPC2020\labels\test")
ROUTER_MODEL_PATH   = Path("optical_rf_stage1.joblib")
ROUTER_META_PATH    = Path("router_metadata.json")
YOLO_MODEL_PATH     = Path(r"C:\Users\pppan\runs\detect\urpc2020_yolov8n-9\weights\best.pt")
WATERNET_WEIGHTS    = Path(r"checkpoints/waternet_best.pth")
UWCNN_WEIGHTS       = Path(r"checkpoints/uwcnn_best.pth")
BASE_EVAL_DIR       = Path(r"C:\internnn\reproducible_eval")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSING_SIZE = (640, 640)
ALPHA_UWCNN = 0.35  # Documented residual scaling parameter

# ================= RESTORATION MODEL DEFINITIONS =================
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
def restore_frame(model, img_bgr, model_type="waternet"):
    orig_h, orig_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, PROCESSING_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t_in = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    if model_type == "waternet":
        t_out = model(t_in).clamp(0.0, 1.0)
    elif model_type == "uwcnn":
        t_out = model(t_in, alpha=ALPHA_UWCNN).clamp(0.0, 1.0)
        
    out_np = t_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    bgr_out = cv2.cvtColor((out_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr_out, (orig_w, orig_h))

# ================= EXECUTION & EVALUATION =================
def evaluate_yolo_on_directory(img_dir: Path, labels_dir: Path, yaml_name: str):
    # Copy labels directly
    for lbl in labels_dir.glob("*.txt"):
        shutil.copy2(lbl, img_dir / lbl.name)
        
    yaml_file = BASE_EVAL_DIR / f"{yaml_name}.yaml"
    with open(yaml_file, "w") as f:
        f.write(f"path: {img_dir.resolve().as_posix()}\ntrain: .\nval: .\nnames:\n  0: holothurian\n  1: echinus\n  2: starfish\n  3: scallops\n")
        
    yolo = YOLO(str(YOLO_MODEL_PATH))
    metrics = yolo.val(data=str(yaml_file), split="val", cache=False, imgsz=640, verbose=False)
    return metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr

def main():
    if BASE_EVAL_DIR.exists():
        shutil.rmtree(BASE_EVAL_DIR)
    BASE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load models
    router = joblib.load(ROUTER_MODEL_PATH)
    with open(ROUTER_META_PATH) as f:
        meta = json.load(f)
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    
    waternet = WaterNet().to(DEVICE).eval()
    waternet.load_state_dict(torch.load(WATERNET_WEIGHTS, map_location=DEVICE))
    
    uwcnn = UWCNN().to(DEVICE).eval()
    uwcnn.load_state_dict(torch.load(UWCNN_WEIGHTS, map_location=DEVICE))
    
    image_paths = sorted([p for p in DETECTION_IMG_DIR.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])
    print(f"[INFO] Manifest established: {len(image_paths)} images found.")
    
    # Setup folders
    dirs = {
        "raw": BASE_EVAL_DIR / "raw",
        "fixed_waternet": BASE_EVAL_DIR / "fixed_waternet",
        "fixed_uwcnn": BASE_EVAL_DIR / "fixed_uwcnn",
        "dynamic": BASE_EVAL_DIR / "dynamic"
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
    
    route_records = {}
    route_counts = {"good": 0, "low_light": 0, "turbid": 0}
    
    # Direct timing accumulators
    t_raw, t_fwn, t_fuw, t_dyn = [], [], [], []
    
    print("[INFO] Processing dataset across configurations and routing dynamically...")
    for p in tqdm(image_paths, desc="Processing Images"):
        img = cv2.imread(str(p))
        if img is None: continue
        
        # 1. Exact raw copy (Identical file bytes, no re-encoding differences)
        shutil.copy2(p, dirs["raw"] / p.name)
        
        # 2. Fixed WaterNet
        t0 = time.perf_counter()
        img_wn = restore_frame(waternet, img, "waternet")
        t_fwn.append(time.perf_counter() - t0)
        cv2.imwrite(str(dirs["fixed_waternet"] / p.name), img_wn)
        
        # 3. Fixed UWCNN
        t0 = time.perf_counter()
        img_uw = restore_frame(uwcnn, img, "uwcnn")
        t_fuw.append(time.perf_counter() - t0)
        cv2.imwrite(str(dirs["fixed_uwcnn"] / p.name), img_uw)
        
        # 4. Proposed Dynamic Pipeline (Feature extraction -> RF prediction -> Conditional execution)
        t0 = time.perf_counter()
        feats = extract_optical_features_from_bgr(img)
        pred_idx = router.predict([feats])[0]
        route = idx_to_class[pred_idx]
        route_counts[route] += 1
        route_records[p.name] = route
        
        if route == "good":
            # Direct copy to bypass restoration and prevent re-encoding artifacts
            shutil.copy2(p, dirs["dynamic"] / p.name)
        elif route == "low_light":
            img_dyn = restore_frame(waternet, img, "waternet")
            cv2.imwrite(str(dirs["dynamic"] / p.name), img_dyn)
        elif route == "turbid":
            img_dyn = restore_frame(uwcnn, img, "uwcnn")
            cv2.imwrite(str(dirs["dynamic"] / p.name), img_dyn)
        t_dyn.append(time.perf_counter() - t0)

    # Dump verified route manifest
    with open("verified_route_manifest.json", "w") as f:
        json.dump(route_records, f, indent=4)
        
    print("\n" + "="*50)
    print("VERIFIED ROUTE DISTRIBUTION (Held-Out RF Model)")
    print("="*50)
    for k, v in route_counts.items():
        print(f"Route: {k:<12} | Count: {v:<5} ({v/len(image_paths)*100:.2f}%)")
    print("="*50 + "\n")
    
    # Run YOLOv8 benchmark
    results = {}
    for name, path_dir in dirs.items():
        print(f"[INFO] Evaluating YOLOv8 on configuration: [{name}]...")
        m50, m, p_score, r_score = evaluate_yolo_on_directory(path_dir, DETECTION_LBL_DIR, name)
        results[name] = {"mAP50": m50, "mAP50-95": m, "Precision": p_score, "Recall": r_score}
        
    print("\n" + "="*70)
    print("FINAL 4-ROW DETECTION COMPARISON TABLE")
    print("="*70)
    print(f"{'Configuration':<25} | {'mAP50':<8} | {'mAP50-95':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-"*70)
    for cfg, m in results.items():
        print(f"{cfg:<25} | {m['mAP50']:<8.4f} | {m['mAP50-95']:<8.4f} | {m['Precision']:<10.4f} | {m['Recall']:<8.4f}")
    print("="*70 + "\n")
    
    # Timing summary (Restoration/Routing compute only)
    print("COMPUTE LATENCY PER FRAME (EXCLUDING DETECTOR FORWARD PASS):")
    print(f"Fixed WaterNet: {np.mean(t_fwn)*1000:.2f} ms")
    print(f"Fixed UWCNN:    {np.mean(t_fuw)*1000:.2f} ms")
    print(f"Dynamic Router: {np.mean(t_dyn)*1000:.2f} ms")

if __name__ == "__main__":
    main()