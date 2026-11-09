# =============================================================================
# # -*- coding: utf-8 -*-
# import os
# import gc
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from ultralytics import YOLO
# from tqdm import tqdm
# 
# # ================= USER PATHS =================
# YOLO_MODEL_PATH = r"C:\final\model\best.pt"
# RAW_VAL_IMAGES = r"C:\internnn\URPC2020\images\val"
# VAL_LABELS_PATH = r"C:\internnn\URPC2020\labels\val"
# 
# WATERNET_OUT_DIR = r"C:\internnn\temp_eval\waternet_all"
# UWCNN_OUT_DIR = r"C:\internnn\temp_eval\uwcnn_all"
# 
# WATERNET_WEIGHTS_PATH = r"checkpoints/waternet_best.pth"
# UWCNN_WEIGHTS_PATH = r"checkpoints/uwcnn_best.pth"
# 
# # ================= HARDWARE SETTINGS =================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 
# if DEVICE.type == "cpu":
#     # Leverage multi-core CPU capabilities
#     torch.set_num_threads(os.cpu_count() or 8)
# 
# # -------------------------------------------------------------------------
# # 1. Model Architectures
# # -------------------------------------------------------------------------
# class WaterNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
# 
#     def forward(self, x):
#         return self.net(x)
# 
# 
# class UWCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
#         
#         self.conv4 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(inplace=True))
#         self.conv5 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(inplace=True))
#         self.conv6 = nn.Sequential(nn.Conv2d(80, 16, 3, padding=1), nn.ReLU(inplace=True))
#         self.conv7 = nn.Sequential(nn.Conv2d(96, 16, 3, padding=1), nn.ReLU(inplace=True))
#         self.conv8 = nn.Sequential(nn.Conv2d(112, 16, 3, padding=1), nn.ReLU(inplace=True))
#         self.conv9 = nn.Sequential(nn.Conv2d(128, 16, 3, padding=1), nn.ReLU(inplace=True))
#         
#         self.conv10 = nn.Conv2d(144, 3, 3, padding=1)
# 
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         
#         cat1 = torch.cat([x1, x2, x3], dim=1)
#         x4 = self.conv4(cat1)
#         
#         cat2 = torch.cat([cat1, x4], dim=1)
#         x5 = self.conv5(cat2)
#         
#         cat3 = torch.cat([cat2, x5], dim=1)
#         x6 = self.conv6(cat3)
#         
#         cat4 = torch.cat([cat3, x6], dim=1)
#         x7 = self.conv7(cat4)
#         
#         cat5 = torch.cat([cat4, x7], dim=1)
#         x8 = self.conv8(cat5)
#         
#         cat6 = torch.cat([cat5, x8], dim=1)
#         x9 = self.conv9(cat6)
#         
#         cat7 = torch.cat([cat6, x9], dim=1)
#         out = self.conv10(cat7)
#         
#         return torch.clamp(x + out, 0.0, 1.0)
# 
# 
# # -------------------------------------------------------------------------
# # 2. Initialization
# # -------------------------------------------------------------------------
# print(f"[INFO] Initializing System on: {DEVICE}")
# 
# waternet_model = WaterNet().to(DEVICE)
# if os.path.exists(WATERNET_WEIGHTS_PATH):
#     waternet_model.load_state_dict(torch.load(WATERNET_WEIGHTS_PATH, map_location=DEVICE))
#     waternet_model.eval()
#     print(f"[SUCCESS] Loaded WaterNet weights from {WATERNET_WEIGHTS_PATH}")
# 
# uwcnn_model = UWCNN().to(DEVICE)
# if os.path.exists(UWCNN_WEIGHTS_PATH):
#     uwcnn_model.load_state_dict(torch.load(UWCNN_WEIGHTS_PATH, map_location=DEVICE))
#     uwcnn_model.eval()
#     print(f"[SUCCESS] Loaded UWCNN weights from {UWCNN_WEIGHTS_PATH}")
# 
# 
# # -------------------------------------------------------------------------
# # 3. Flexible Single-Image Streaming Engine (Handles Any Resolution)
# # -------------------------------------------------------------------------
# @torch.inference_mode()
# def safe_enhance_dataset(model, img_paths, output_dir):
#     """Processes images individually to allow arbitrary native dimensions without errors"""
#     os.makedirs(output_dir, exist_ok=True)
#     
#     for path in tqdm(img_paths, desc="Enhancing Images"):
#         filename = os.path.basename(path)
#         out_path = os.path.join(output_dir, filename)
#         
#         # Resume capability: Skip if already processed
#         if os.path.exists(out_path):
#             continue
#             
#         img = cv2.imread(path)
#         if img is None: 
#             continue
#             
#         # 1. BGR -> RGB and Normalize [0, 1]
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
#         tensor_img = tensor_img.unsqueeze(0).to(DEVICE)  # Shape: (1, 3, H, W)
#         
#         # 2. Forward pass at true native resolution
#         enhanced_tensor = model(tensor_img).clamp(0.0, 1.0)
#         
#         # 3. Postprocess and Save Immediately
#         out_tensor = enhanced_tensor.squeeze(0).cpu()
#         img_np = (out_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
#         bgr_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(out_path, bgr_np)
#         
#         # Memory Cleanup
#         del tensor_img, enhanced_tensor, out_tensor
# 
# 
# # -------------------------------------------------------------------------
# # 4. Pipeline Runner
# # -------------------------------------------------------------------------
# def process_and_evaluate(model, output_dir, pipeline_name):
#     image_files = [
#         os.path.join(RAW_VAL_IMAGES, f) 
#         for f in os.listdir(RAW_VAL_IMAGES) 
#         if f.lower().endswith(('.jpg', '.png', '.jpeg'))
#     ]
#     
#     print(f"\n[INFO] Starting {pipeline_name} enhancement on {len(image_files)} images...")
#     safe_enhance_dataset(model, image_files, output_dir)
#         
#     print(f"\n[INFO] Evaluating YOLOv8 on {pipeline_name} set...")
#     yolo_model = YOLO(YOLO_MODEL_PATH)
#     metrics = yolo_model.val(data={'val': output_dir, 'labels': VAL_LABELS_PATH})
#     
#     print("\n" + "="*40)
#     print(f"   RESULTS FOR {pipeline_name}")
#     print("="*40)
#     print(f"mAP50:     {metrics.box.map50:.4f}")
#     print(f"mAP50-95:  {metrics.box.map:.4f}")
#     print(f"Precision: {metrics.box.mp:.4f}")
#     print(f"Recall:    {metrics.box.mr:.4f}")
#     print("="*40)
# 
# 
# if __name__ == "__main__":
#     # Run WaterNet-All Benchmark
#     process_and_evaluate(waternet_model, WATERNET_OUT_DIR, "WaterNet-All")
# 
#     # Run UWCNN-All Benchmark
#     process_and_evaluate(uwcnn_model, UWCNN_OUT_DIR, "UWCNN-All")
# =============================================================================
# =============================================================================
# import os
# import yaml
# from ultralytics import YOLO
# 
# # ================= USER PATHS =================
# YOLO_MODEL_PATH = r"C:\final\model\best.pt"
# 
# WATERNET_OUT_DIR = r"C:\internnn\temp_eval\waternet_all"
# UWCNN_OUT_DIR = r"C:\internnn\temp_eval\uwcnn_all"
# 
# # ================= FIXED YAML GENERATOR =================
# def create_dataset_yaml(images_dir, yaml_path="dataset_eval.yaml"):
#     """
#     Creates a standard Ultralytics YAML pointing to the image folder.
#     Ultralytics automatically matches 'images' with the adjacent 'labels' folder.
#     """
#     abs_images = os.path.abspath(images_dir)
#     
#     dataset_dict = {
#         'path': os.path.dirname(abs_images),
#         'train': abs_images,
#         'val': abs_images,
#         'names': {
#             0: 'holothurian',
#             1: 'echinus',
#             2: 'starfish',
#             3: 'scallops'
#         }
#     }
#     
#     with open(yaml_path, 'w') as f:
#         yaml.dump(dataset_dict, f)
#         
#     return yaml_path
# 
# 
# def evaluate_enhanced_folder(enhanced_dir, pipeline_name):
#     print(f"\n[INFO] Running YOLOv8 Evaluation on {pipeline_name} dataset...")
#     
#     yaml_file = f"eval_{pipeline_name}.yaml"
#     create_dataset_yaml(enhanced_dir, yaml_file)
#     
#     # Load model and run validation
#     model = YOLO(YOLO_MODEL_PATH)
#     
#     # Run validation using standard Ultralytics parameters
#     metrics = model.val(data=yaml_file, split='val')
#     
#     # Display formal results for IEEE tables
#     print("\n" + "="*45)
#     print(f"   EVALUATION RESULTS FOR {pipeline_name}")
#     print("="*45)
#     print(f"Overall mAP50:     {metrics.box.map50:.4f}")
#     print(f"Overall mAP50-95:  {metrics.box.map:.4f}")
#     print(f"Overall Precision: {metrics.box.mp:.4f}")
#     print(f"Overall Recall:    {metrics.box.mr:.4f}")
#     print("="*45)
#     
#     # Per-Class Breakdown
#     if hasattr(metrics.box, 'map50_per_class'):
#         classes = ['holothurian', 'echinus', 'starfish', 'scallops']
#         print("\n--- Per-Class mAP50 Breakdown ---")
#         for idx, cls_name in enumerate(classes):
#             if idx < len(metrics.box.map50_per_class):
#                 print(f"{cls_name:<15}: {metrics.box.map50_per_class[idx]:.4f}")
#         print("="*45)
# 
#     # Clean up temporary file
#     if os.path.exists(yaml_file):
#         os.remove(yaml_file)
# 
# 
# if __name__ == "__main__":
#     # 1. Evaluate WaterNet-All
#     if os.path.exists(WATERNET_OUT_DIR) and len(os.listdir(WATERNET_OUT_DIR)) > 0:
#         evaluate_enhanced_folder(WATERNET_OUT_DIR, "WaterNet-All")
#         
#     # 2. Evaluate UWCNN-All (when ready)
#     if os.path.exists(UWCNN_OUT_DIR) and len(os.listdir(UWCNN_OUT_DIR)) > 0:
#         evaluate_enhanced_folder(UWCNN_OUT_DIR, "UWCNN-All")
# =============================================================================

# =============================================================================
# 
# import shutil
# import os
# 
# # Copy your original labels into the temp_eval folder
# src_labels = r"C:\internnn\URPC2020\labels\val"
# dst_labels = r"C:\internnn\temp_eval\labels\val"
# 
# os.makedirs(dst_labels, exist_ok=True)
# shutil.copytree(src_labels, dst_labels, dirs_exist_ok=True)
# print("[SUCCESS] Ground-truth labels copied to temp_eval/labels/val")
# =============================================================================
# =============================================================================
# 
# import os
# 
# cache_files = [
#     r"C:\internnn\temp_eval\waternet_all.cache",
#     r"C:\internnn\temp_eval\waternet_all\labels.cache",
#     r"C:\internnn\temp_eval\labels\val.cache"
# ]
# 
# for c in cache_files:
#     if os.path.exists(c):
#         os.remove(c)
#         print(f"[REMOVED] Stale cache: {c}")
# =============================================================================
# =============================================================================
# 
# import os
# import shutil
# 
# src_labels_dir = r"C:\internnn\URPC2020\labels\val"
# target_img_dir = r"C:\internnn\temp_eval\waternet_all"
# 
# label_files = [f for f in os.listdir(src_labels_dir) if f.endswith('.txt')]
# print(f"[INFO] Copying {len(label_files)} label files directly into {target_img_dir}...")
# 
# for lbl in label_files:
#     src_path = os.path.join(src_labels_dir, lbl)
#     dst_path = os.path.join(target_img_dir, lbl)
#     shutil.copy2(src_path, dst_path)
# 
# print("[SUCCESS] All labels copied directly into the image folder!")
# =============================================================================
# =============================================================================
# 
# import os
# import yaml
# from ultralytics import YOLO
# 
# YOLO_MODEL_PATH = r"C:\final\model\best.pt"
# WATERNET_OUT_DIR = r"C:\internnn\temp_eval\waternet_all"
# 
# def evaluate_waternet_direct():
#     abs_path = os.path.abspath(WATERNET_OUT_DIR)
#     yaml_path = "direct_val.yaml"
#     
#     # Simple, strict Ultralytics YAML structure
#     dataset_dict = {
#         'path': abs_path,
#         'train': abs_path,
#         'val': abs_path,
#         'names': {
#             0: 'holothurian',
#             1: 'echinus',
#             2: 'starfish',
#             3: 'scallops'
#         }
#     }
#     
#     with open(yaml_path, 'w') as f:
#         yaml.dump(dataset_dict, f)
#         
#     print("\n[INFO] Running YOLOv8 Evaluation on WaterNet-All dataset...")
#     model = YOLO(YOLO_MODEL_PATH)
#     
#     # Force fresh scan (rect=False, cache=False)
#     metrics = model.val(data=yaml_path, split='val', cache=False)
#     
#     print("\n" + "="*45)
#     print("   FINAL EVALUATION RESULTS FOR WaterNet-All")
#     print("="*45)
#     print(f"Overall mAP50:     {metrics.box.map50:.4f}")
#     print(f"Overall mAP50-95:  {metrics.box.map:.4f}")
#     print(f"Overall Precision: {metrics.box.mp:.4f}")
#     print(f"Overall Recall:    {metrics.box.mr:.4f}")
#     print("="*45)
# 
#     if os.path.exists(yaml_path):
#         os.remove(yaml_path)
# 
# if __name__ == "__main__":
#     evaluate_waternet_direct()
# =============================================================================


# -*- coding: utf-8 -*-
import os
import gc
import cv2
import shutil
import numpy as np
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from tqdm import tqdm

# ================= USER PATHS =================
YOLO_MODEL_PATH = r"C:\final\model\best.pt"
RAW_VAL_IMAGES = r"C:\internnn\URPC2020\images\val"
LABELS_SRC_DIR = r"C:\internnn\URPC2020\labels\val"

UWCNN_OUT_DIR = r"C:\internnn\temp_eval\uwcnn_all"
UWCNN_WEIGHTS_PATH = r"checkpoints/uwcnn_best.pth"

# Target resolution for CPU speedup (Prevents WinError 1450)
PROCESSING_SIZE = (640, 640)  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)

# -------------------------------------------------------------------------
# 1. UWCNN Architecture
# -------------------------------------------------------------------------
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

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        cat1 = torch.cat([x1, x2, x3], dim=1)
        x4 = self.conv4(cat1)
        
        cat2 = torch.cat([cat1, x4], dim=1)
        x5 = self.conv5(cat2)
        
        cat3 = torch.cat([cat2, x5], dim=1)
        x6 = self.conv6(cat3)
        
        cat4 = torch.cat([cat3, x6], dim=1)
        x7 = self.conv7(cat4)
        
        cat5 = torch.cat([cat4, x7], dim=1)
        x8 = self.conv8(cat5)
        
        cat6 = torch.cat([cat5, x8], dim=1)
        x9 = self.conv9(cat6)
        
        cat7 = torch.cat([cat6, x9], dim=1)
        out = self.conv10(cat7)
        
        return torch.clamp(x + out, 0.0, 1.0)

# -------------------------------------------------------------------------
# 2. Load Weights
# -------------------------------------------------------------------------
print(f"[INFO] Initializing UWCNN on device: {DEVICE}")
uwcnn_model = UWCNN().to(DEVICE)

if os.path.exists(UWCNN_WEIGHTS_PATH):
    uwcnn_model.load_state_dict(torch.load(UWCNN_WEIGHTS_PATH, map_location=DEVICE))
    uwcnn_model.eval()
    print(f"[SUCCESS] Loaded UWCNN weights from {UWCNN_WEIGHTS_PATH}")
else:
    raise FileNotFoundError(f"UWCNN weights not found at {UWCNN_WEIGHTS_PATH}")

# -------------------------------------------------------------------------
# 3. Fast Resized Enhancement Stream
# -------------------------------------------------------------------------
@torch.inference_mode()
def enhance_dataset_uwcnn_fast(model, img_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for path in tqdm(img_paths, desc="Fast UWCNN Enhancement"):
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        
        # Resume: Skip images already generated
        if os.path.exists(out_path):
            continue
            
        img = cv2.imread(path)
        if img is None: 
            continue
            
        orig_h, orig_w = img.shape[:2]
        
        # 1. Resize for fast CPU forward pass
        img_resized = cv2.resize(img, PROCESSING_SIZE)
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor_img = tensor_img.unsqueeze(0).to(DEVICE)
        
        # 2. Run UWCNN
        enhanced_tensor = model(tensor_img).clamp(0.0, 1.0)
        
        # 3. Restore to original native dimension & save
        out_tensor = enhanced_tensor.squeeze(0).cpu()
        img_np = (out_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        bgr_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        bgr_restored = cv2.resize(bgr_np, (orig_w, orig_h))
        
        cv2.imwrite(out_path, bgr_restored)
        
        del tensor_img, enhanced_tensor, out_tensor

# -------------------------------------------------------------------------
# 4. Sync Labels & Run Validation
# -------------------------------------------------------------------------
def run_uwcnn_benchmark():
    image_files = [
        os.path.join(RAW_VAL_IMAGES, f) 
        for f in os.listdir(RAW_VAL_IMAGES) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    # 1. Enhance
    print(f"\n[INFO] Resuming UWCNN enhancement on remaining images...")
    enhance_dataset_uwcnn_fast(uwcnn_model, image_files, UWCNN_OUT_DIR)
    
    # 2. Copy labels into output directory alongside images
    print("[INFO] Syncing ground-truth label annotations...")
    label_files = [f for f in os.listdir(LABELS_SRC_DIR) if f.endswith('.txt')]
    for lbl in label_files:
        dst_p = os.path.join(UWCNN_OUT_DIR, lbl)
        if not os.path.exists(dst_p):
            shutil.copy2(os.path.join(LABELS_SRC_DIR, lbl), dst_p)

    # 3. Create Dataset YAML
    abs_path = os.path.abspath(UWCNN_OUT_DIR)
    yaml_path = "uwcnn_eval.yaml"
    dataset_dict = {
        'path': abs_path,
        'train': abs_path,
        'val': abs_path,
        'names': {0: 'holothurian', 1: 'echinus', 2: 'starfish', 3: 'scallops'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_dict, f)

    # 4. Validate
    print(f"\n[INFO] Running YOLOv8 Evaluation on UWCNN-All dataset...")
    model = YOLO(YOLO_MODEL_PATH)
    metrics = model.val(data=yaml_path, split='val', cache=False)
    
    print("\n" + "="*45)
    print("   FINAL EVALUATION RESULTS FOR UWCNN-All")
    print("="*45)
    print(f"Overall mAP50:     {metrics.box.map50:.4f}")
    print(f"Overall mAP50-95:  {metrics.box.map:.4f}")
    print(f"Overall Precision: {metrics.box.mp:.4f}")
    print(f"Overall Recall:    {metrics.box.mr:.4f}")
    print("="*45)

    if os.path.exists(yaml_path):
        os.remove(yaml_path)

if __name__ == "__main__":
    run_uwcnn_benchmark()