# common_features.py
import cv2
import numpy as np

FEATURE_NAMES = [
    "Laplacian Blur Variance",
    "L* Luminance Mean",
    "L* Luminance Std",
    "a* Green-Red Cast",
    "b* Blue-Yellow Cast",
    "Saturation Mean",
    "RMS Contrast",
    "Dark Channel Proxy",
    "Red/Blue Attenuation"
]

def extract_optical_features_from_bgr(img_bgr: np.ndarray) -> list:
    """
    Extracts 9 optics-inspired features from an in-memory BGR image.
    Strictly resizes to 224x224 to standardize scale-dependent derivatives.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid image array passed to feature extractor.")
        
    img_resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    hsv  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # 1. Laplacian blur variance (high-frequency edge energy)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. L* luminance mean (exposure level)
    l_mean = float(lab[:, :, 0].mean())
    
    # 3. L* luminance standard deviation (dynamic range)
    l_std = float(lab[:, :, 0].std())
    
    # 4. a* chromaticity mean (green-red axis)
    a_mean = float(lab[:, :, 1].mean())
    
    # 5. b* chromaticity mean (blue-yellow axis)
    b_mean = float(lab[:, :, 2].mean())
    
    # 6. HSV saturation mean (color concentration)
    saturation = float(hsv[:, :, 1].mean())
    
    # 7. Gray RMS contrast (overall intensity dispersion)
    contrast = float(gray.std())
    
    # 8. Dark channel proxy (mean across minimum color channels)
    dark_prior = float(np.min(img_resized, axis=2).mean())
    
    # 9. Red/Blue attenuation ratio (safeguarded against division by zero)
    attenuation_r = float(img_resized[:, :, 2].mean() / (img_resized[:, :, 0].mean() + 1e-5))
    
    return [
        laplacian_var, l_mean, l_std, a_mean, b_mean, 
        saturation, contrast, dark_prior, attenuation_r
    ]

def extract_optical_features(img_path: str) -> list:
    img = cv2.imread(img_path)
    if img is None:
        return None
    return extract_optical_features_from_bgr(img)