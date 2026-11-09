# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= USER PATHS =================
RAW_DIR = r"C:\internnn\URPC2020\images\val"
WATERNET_DIR = r"C:\internnn\temp_eval\waternet_all"
UWCNN_DIR = r"C:\internnn\temp_eval\uwcnn_all"

# -------------------------------------------------------------------------
# 1. Metric Calculation Functions
# -------------------------------------------------------------------------
def calculate_uciqe(img):
    """
    Computes Underwater Color Image Quality Evaluation (UCIQE).
    Measures chroma variance, saturation, and contrast in CIELab space.
    Higher value = Better color balance and contrast.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Chroma variance
    chroma = np.sqrt(a.astype(np.float64)**2 + b.astype(np.float64)**2)
    var_chroma = np.var(chroma)
    
    # Contrast of Luminance
    con_lum = np.std(l.astype(np.float64))
    
    # Saturation
    saturation = chroma / (l.astype(np.float64) + 1e-5)
    mean_sat = np.mean(saturation)
    
    # Standard UCIQE Coefficients
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return c1 * var_chroma + c2 * con_lum + c3 * mean_sat


def calculate_uiqm(img):
    """
    Computes Underwater Image Quality Measure (UIQM).
    Combines UICM (colorfulness), UISM (sharpness), and UIConM (contrast).
    Higher value = Better visual clarity and naturalness.
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    # 1. UICM (Colorfulness Measure)
    rg = r - g
    yb = 0.5 * (r + g) - b
    mean_rg, std_rg = np.mean(rg), np.std(rg)
    mean_yb, std_yb = np.mean(yb), np.std(yb)
    uicm = -0.0268 * np.sqrt(mean_rg**2 + mean_yb**2) + 0.1586 * np.sqrt(std_rg**2 + std_yb**2)
    
    # 2. UISM (Sharpness Measure via Sobel Operator)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    uism = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    # 3. UIConM (Contrast Measure)
    uiconm = np.std(gray)
    
    # Linear Combination Coefficients for Underwater Scenery
    c1, c2, c3 = 0.0282, 0.2953, 0.6765
    return c1 * uicm + c2 * uism + c3 * uiconm

# -------------------------------------------------------------------------
# 2. Benchmark Execution
# -------------------------------------------------------------------------
def evaluate_quality_metrics():
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:200]  # Sample 200 frames for speed
    
    raw_uciqe, raw_uiqm = [], []
    wn_uciqe, wn_uiqm = [], []
    uw_uciqe, uw_uiqm = [], []
    
    print(f"[INFO] Computing UIQM and UCIQE across {len(files)} sample frames...")
    
    for filename in tqdm(files, desc="Evaluating Image Quality"):
        p_raw = os.path.join(RAW_DIR, filename)
        p_wn = os.path.join(WATERNET_DIR, filename)
        p_uw = os.path.join(UWCNN_DIR, filename)
        
        img_raw = cv2.imread(p_raw)
        if img_raw is not None:
            raw_uciqe.append(calculate_uciqe(img_raw))
            raw_uiqm.append(calculate_uiqm(img_raw))
            
        if os.path.exists(p_wn):
            img_wn = cv2.imread(p_wn)
            if img_wn is not None:
                wn_uciqe.append(calculate_uciqe(img_wn))
                wn_uiqm.append(calculate_uiqm(img_wn))
                
        if os.path.exists(p_uw):
            img_uw = cv2.imread(p_uw)
            if img_uw is not None:
                uw_uciqe.append(calculate_uciqe(img_uw))
                uw_uiqm.append(calculate_uiqm(img_uw))

    print("\n" + "="*50)
    print("   NON-REFERENCE UNDERWATER IMAGE QUALITY BENCHMARK")
    print("="*50)
    print(f"Raw Images:     UCIQE = {np.mean(raw_uciqe):.4f}  |  UIQM = {np.mean(raw_uiqm):.4f}")
    if wn_uciqe:
        print(f"WaterNet Set:   UCIQE = {np.mean(wn_uciqe):.4f}  |  UIQM = {np.mean(wn_uiqm):.4f}  (Gain: +{((np.mean(wn_uiqm)-np.mean(raw_uiqm))/np.mean(raw_uiqm))*100:.1f}% UIQM)")
    if uw_uciqe:
        print(f"UWCNN Set:      UCIQE = {np.mean(uw_uciqe):.4f}  |  UIQM = {np.mean(uw_uiqm):.4f}  (Gain: +{((np.mean(uw_uiqm)-np.mean(raw_uiqm))/np.mean(raw_uiqm))*100:.1f}% UIQM)")
    print("="*50)

if __name__ == "__main__":
    evaluate_quality_metrics()