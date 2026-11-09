# =============================================================================
# import os
# 
# ROOT = r"C:\Users\pppan\OneDrive\Desktop\internnn\URPC2020"
# 
# for split in ["train", "val", "test"]:
#     img_dir = os.path.join(ROOT, "images", split)
#     lbl_dir = os.path.join(ROOT, "labels", split)
# 
#     print(f"\n=== {split.upper()} ===")
#     print("images:", len(os.listdir(img_dir)) if os.path.isdir(img_dir) else "missing")
#     print("labels:", len(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else "missing")
# 
#     imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.isdir(img_dir) else []
#     lbls = sorted([f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]) if os.path.isdir(lbl_dir) else []
# 
#     print("first images:", imgs[:5])
#     print("first labels:", lbls[:5])
# 
#     missing = [os.path.splitext(f)[0] for f in imgs if not os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))]
#     print("missing label count:", len(missing))
#     if missing[:5]:
#         print("examples:", missing[:5])
# ===========================================================================
# =============================================================================
# # =============================================================================
import torch
from ultralytics import YOLO

device = "0" if torch.cuda.is_available() else "cpu"
print("selected device:", device)

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\internnn\URPC2020\data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=device,
    name="urpc2020_yolov8n"
)

print("Training finished.")
# =============================================================================
# =============================================================================
# # =============================================================================
# import torch
# 
# print("torch:", torch.__version__)
# print("cuda available:", torch.cuda.is_available())
# print("cuda device count:", torch.cuda.device_count())
# 
# if torch.cuda.is_available():
#     print("gpu name:", torch.cuda.get_device_name(0))
#     device = "0"
# else:
#     print("using cpu")
#     device = "cpu"
# 
# print("selected device:", device)
# =============================================================================from pathlib import Path
# =============================================================================
# from pathlib import Path
# root_dir = Path(r"C:\internnn\URPC2020")
# targets = {"images", "labels", "train", "val", "test", "valid"}
# 
# if root_dir.exists():
#     print(f"\nROOT: {root_dir}")
#     for p in root_dir.rglob("*"):
#         if p.is_dir() and p.name in targets:
#             print(p)
# else:
#     print(f"Directory does not exist: {root_dir}")
# =============================================================================
