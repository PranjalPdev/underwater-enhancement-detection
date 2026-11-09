import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from tqdm import tqdm

# --- CONFIGURATION ---
SRC_RAW_DIR   = Path(r"C:\internnn\URPC2020\images\test")
LBL_DIR       = Path(r"C:\internnn\URPC2020\labels\test")
MANIFEST_PATH = Path("verified_route_manifest.json")
EVAL_BASE     = Path(r"C:\internnn\route_eval_clean")

WATERNET_WEIGHTS = Path(r"checkpoints/waternet_best.pth")
UWCNN_WEIGHTS    = Path(r"checkpoints/uwcnn_best.pth")
YOLO_WT          = Path(r"C:\Users\pppan\runs\detect\urpc2020_yolov8n-9\weights\best.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSING_SIZE = (640, 640)
ALPHA_UWCNN = 0.35

# --- ARCHITECTURES ---
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

def main():
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Could not find {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    if EVAL_BASE.exists():
        shutil.rmtree(EVAL_BASE)
    EVAL_BASE.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading restoration networks...")
    waternet = WaterNet().to(DEVICE).eval()
    waternet.load_state_dict(torch.load(WATERNET_WEIGHTS, map_location=DEVICE))

    uwcnn = UWCNN().to(DEVICE).eval()
    uwcnn.load_state_dict(torch.load(UWCNN_WEIGHTS, map_location=DEVICE))

    unique_routes = sorted(list(set(manifest.values())))
    route_dirs = {}

    for r in unique_routes:
        r_raw = EVAL_BASE / f"{r}_raw"
        r_dyn = EVAL_BASE / f"{r}_dyn"
        r_raw.mkdir(parents=True, exist_ok=True)
        r_dyn.mkdir(parents=True, exist_ok=True)
        route_dirs[r] = {"raw": r_raw, "dyn": r_dyn}

    print(f"[INFO] Populating subsets across {len(manifest)} manifest entries...")
    for fname, route in tqdm(manifest.items(), desc="Generating Subsets"):
        src_img = SRC_RAW_DIR / fname
        if not src_img.exists():
            continue

        lbl_file = LBL_DIR / (Path(fname).stem + ".txt")
        raw_dest = route_dirs[route]["raw"] / fname
        dyn_dest = route_dirs[route]["dyn"] / fname

        # Copy raw
        shutil.copy2(src_img, raw_dest)
        if lbl_file.exists():
            shutil.copy2(lbl_file, route_dirs[route]["raw"] / lbl_file.name)
            shutil.copy2(lbl_file, route_dirs[route]["dyn"] / lbl_file.name)

        # Generate or route dynamic output directly
        if route == "good":
            shutil.copy2(src_img, dyn_dest)
        elif route == "low_light":
            img = cv2.imread(str(src_img))
            res = restore_frame(waternet, img, "waternet")
            cv2.imwrite(str(dyn_dest), res)
        elif route == "turbid":
            img = cv2.imread(str(src_img))
            res = restore_frame(uwcnn, img, "uwcnn")
            cv2.imwrite(str(dyn_dest), res)

    print("\n[INFO] Running YOLOv8 route-wise evaluations...")
    yolo = YOLO(str(YOLO_WT))

    print("\n" + "="*80)
    print("VERIFIED ROUTE-WISE ABLATION TABLE (ALL IMAGES)")
    print("="*80)
    print(f"{'Route':<14} | {'Images':<8} | {'Raw mAP50':<12} | {'Pipeline mAP50':<16} | {'Delta':<8}")
    print("-"*80)

    for r in unique_routes:
        d_raw = route_dirs[r]["raw"]
        d_dyn = route_dirs[r]["dyn"]
        
        n_imgs = len(list(d_raw.glob("*.jpg"))) + len(list(d_raw.glob("*.png")))
        if n_imgs == 0:
            continue

        y_raw = EVAL_BASE / f"{r}_raw.yaml"
        y_dyn = EVAL_BASE / f"{r}_dyn.yaml"

        for y_path, d_path in [(y_raw, d_raw), (y_dyn, d_dyn)]:
            with open(y_path, "w") as fp:
                fp.write(f"path: {d_path.resolve().as_posix()}\ntrain: .\nval: .\nnames:\n  0: holothurian\n  1: echinus\n  2: starfish\n  3: scallops\n")

        m_raw = yolo.val(data=str(y_raw), split="val", cache=False, imgsz=640, verbose=False).box.map50
        m_dyn = yolo.val(data=str(y_dyn), split="val", cache=False, imgsz=640, verbose=False).box.map50
        delta = m_dyn - m_raw

        print(f"{r:<14} | {n_imgs:<8} | {m_raw:<12.4f} | {m_dyn:<16.4f} | {delta:<+8.4f}")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()