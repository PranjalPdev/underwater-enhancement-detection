# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:02:04 2026

@author: pppan
"""

# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# ================= USER PATHS =================
CLASSIFIER_PATH = r"C:\internnn\efficientnet_classifier.keras"
YOLO_MODEL_PATH = r"C:\final\model\best.pt"
RAW_VAL_IMAGES = r"C:\internnn\URPC2020\images\val"

WATERNET_WEIGHTS = r"checkpoints/waternet_best.pth"
UWCNN_WEIGHTS = r"checkpoints/uwcnn_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
classifier = load_model(CLASSIFIER_PATH)
yolo = YOLO(YOLO_MODEL_PATH)

class WaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class UWCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Conv2d(80, 16, 3, padding=1), nn.ReLU(True))
        self.conv7 = nn.Sequential(nn.Conv2d(96, 16, 3, padding=1), nn.ReLU(True))
        self.conv8 = nn.Sequential(nn.Conv2d(112, 16, 3, padding=1), nn.ReLU(True))
        self.conv9 = nn.Sequential(nn.Conv2d(128, 16, 3, padding=1), nn.ReLU(True))
        self.conv10 = nn.Conv2d(144, 3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x); x2 = self.conv2(x1); x3 = self.conv3(x2)
        cat1 = torch.cat([x1, x2, x3], dim=1); x4 = self.conv4(cat1)
        cat2 = torch.cat([cat1, x4], dim=1); x5 = self.conv5(cat2)
        cat3 = torch.cat([cat2, x5], dim=1); x6 = self.conv6(cat3)
        cat4 = torch.cat([cat3, x6], dim=1); x7 = self.conv7(cat4)
        cat5 = torch.cat([cat4, x7], dim=1); x8 = self.conv8(cat5)
        cat6 = torch.cat([cat5, x8], dim=1); x9 = self.conv9(cat6)
        cat7 = torch.cat([cat6, x9], dim=1); out = self.conv10(cat7)
        return torch.clamp(x + out, 0.0, 1.0)

wn_model = WaterNet().to(DEVICE); wn_model.load_state_dict(torch.load(WATERNET_WEIGHTS, map_location=DEVICE)); wn_model.eval()
uw_model = UWCNN().to(DEVICE); uw_model.load_state_dict(torch.load(UWCNN_WEIGHTS, map_location=DEVICE)); uw_model.eval()

# Warmup pass
dummy = np.zeros((224, 224, 3), dtype=np.uint8)
classifier.predict(np.expand_dims(dummy, 0), verbose=0)

files = [os.path.join(RAW_VAL_IMAGES, f) for f in os.listdir(RAW_VAL_IMAGES) if f.endswith(('.jpg', '.png'))][:100]

t_cls, t_wn, t_uw, t_yolo = [], [], [], []

print("[INFO] Benchmarking Latency across 100 frames...")
for p in files:
    img = cv2.imread(p)
    
    # Stage 1: Classifier
    t0 = time.time()
    rgb_cls = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    classifier.predict(np.expand_dims(rgb_cls, axis=0), verbose=0)
    t_cls.append((time.time() - t0) * 1000)
    
    # Stage 2: Restoration
    tensor_img = torch.from_numpy(cv2.resize(img, (640, 640))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    t0 = time.time()
    with torch.no_grad(): wn_model(tensor_img)
    t_wn.append((time.time() - t0) * 1000)
    
    t0 = time.time()
    with torch.no_grad(): uw_model(tensor_img)
    t_uw.append((time.time() - t0) * 1000)

    # Stage 3: YOLOv8
    t0 = time.time()
    yolo.predict(img, verbose=False)
    t_yolo.append((time.time() - t0) * 1000)

avg_cls = np.mean(t_cls)
avg_wn = np.mean(t_wn)
avg_uw = np.mean(t_uw)
avg_yolo = np.mean(t_yolo)

# Calculate weighted total pipeline latency based on your exact routing breakdown:
# Good (Bypass): 54.6%, Low-Light (WaterNet): 39.1%, Turbid (UWCNN): 6.3%
avg_pipeline_latency = avg_cls + (0.546 * 0) + (0.391 * avg_wn) + (0.063 * avg_uw) + avg_yolo
pipeline_fps = 1000.0 / avg_pipeline_latency

# Calculate fixed baseline Latency for comparison
fixed_wn_latency = avg_cls + avg_wn + avg_yolo
fixed_uw_latency = avg_cls + avg_uw + avg_yolo

print("\n" + "="*50)
print("       LATENCY AND THROUGHOUT BENCHMARK RESULTS")
print("="*50)
print(f"Stage 1 Classifier Latency:  {avg_cls:.2f} ms")
print(f"Stage 2 WaterNet Latency:    {avg_wn:.2f} ms")
print(f"Stage 2 UWCNN Latency:       {avg_uw:.2f} ms")
print(f"Stage 3 YOLOv8 Latency:      {avg_yolo:.2f} ms")
print("-" * 50)
print(f"Proposed Pipeline Latency:   {avg_pipeline_latency:.2f} ms per frame")
print(f"Proposed Pipeline Speed:     {pipeline_fps:.2f} FPS")
print(f"Fixed WaterNet-All Speed:    {1000.0/fixed_wn_latency:.2f} FPS")
print(f"Fixed UWCNN-All Speed:       {1000.0/fixed_uw_latency:.2f} FPS")
print("="*50)