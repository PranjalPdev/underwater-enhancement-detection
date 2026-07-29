import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from torchvision import transforms
from ultralytics import YOLO

# ==========================================
# Paths & Configurations
# ==========================================
BASE = Path(r"C:\final")
INPUT_IMAGE = BASE / "input" / "demo.jpg"  # Update filename if using demo.webp/demo.jpg
OUTPUT_DIR = BASE / "output"
MODELS_DIR = BASE / "model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TensorFlow / Keras Classifier Model & Class Indices
CLASSIFIER_WEIGHTS = MODELS_DIR / "efficientnet_classifier.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# PyTorch Enhancement Checkpoints & YOLO
WATERNET_WEIGHTS = MODELS_DIR / "waternet_best.pth"
UWCNN_WEIGHTS = MODELS_DIR / "uwcnn_best.pth"
YOLO_WEIGHTS = MODELS_DIR / "best.pt"  # Best weights from your trained YOLO model

CLASSIFIER_OUT = OUTPUT_DIR / "classifier_result.txt"
WATERNET_OUT = OUTPUT_DIR / "waternet_enhanced.jpg"
UWCNN_OUT = OUTPUT_DIR / "uwcnn_enhanced.jpg"
FINAL_ENHANCED_OUT = OUTPUT_DIR / "selected_enhanced.jpg"
YOLO_PROJECT = OUTPUT_DIR / "yolo_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference transform (converts to tensor without resizing/cropping)
infer_tfm = transforms.Compose([
    transforms.ToTensor()
])

# ==========================================
# PyTorch Enhancement Architectures
# ==========================================
class WaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


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

# ==========================================
# Core Processing Functions
# ==========================================
def run_classifier():
    """Predicts quality category using TensorFlow EfficientNet classifier."""
    if not CLASSIFIER_WEIGHTS.exists():
        print(f"Warning: Classifier weights not found at {CLASSIFIER_WEIGHTS}. Defaulting to 'good'.")
        return "good"

    model = tf.keras.models.load_model(CLASSIFIER_WEIGHTS)
    
    # Get expected input dimension from Keras model
    input_shape = model.input_shape[1:3]
    target_size = input_shape if None not in input_shape else (224, 224)

    img = Image.open(INPUT_IMAGE).convert("RGB").resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)
    idx = int(np.argmax(predictions, axis=1)[0])

    label_map = {}
    if CLASS_INDICES_PATH.exists():
        try:
            with open(CLASS_INDICES_PATH, "r") as f:
                class_indices = json.load(f)
            label_map = {int(v): k for k, v in class_indices.items()}
        except Exception as e:
            print(f"Could not parse class_indices.json: {e}")

    if not label_map:
        label_map = {0: "good", 1: "low_light", 2: "turbid"}

    label = label_map.get(idx, "good")
    CLASSIFIER_OUT.write_text(f"image={INPUT_IMAGE.name}\nlabel={label}\nclass_index={idx}\n")
    return label


def run_enhancement_model(model_class, weights_path, out_path):
    """Executes PyTorch enhancement model on full native resolution image."""
    img = Image.open(INPUT_IMAGE).convert("RGB")
    
    # Process image at full native resolution
    tensor_in = infer_tfm(img).unsqueeze(0).to(DEVICE)

    model = model_class().to(DEVICE)
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"Successfully loaded model weights from: {weights_path.name}")
    else:
        print(f"Warning: Model weights missing at {weights_path}. Running with initialized weights.")

    model.eval()
    with torch.no_grad():
        enhanced_tensor = model(tensor_in)

    # Convert back to uint8 PIL image
    enhanced_tensor = enhanced_tensor.squeeze(0).cpu().clamp(0, 1)
    enhanced_np = (enhanced_tensor.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    Image.fromarray(enhanced_np).save(out_path)
    return out_path


def run_yolo(source_img):
    """Executes YOLOv8 object detection on the routed image."""
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing YOLO weights file at {YOLO_WEIGHTS}")
        
    model = YOLO(str(YOLO_WEIGHTS))
    model.predict(
        source=str(source_img),
        save=True,
        project=str(YOLO_PROJECT),
        name="pred",
        exist_ok=True,
        conf=0.25,
        imgsz=640
    )

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Input image missing at: {INPUT_IMAGE}")

    print("Step 1: Running Image Quality Analysis...")
    label = run_classifier()
    print(f"--> Classification Label Detected: '{label}'")

    print("\nStep 2: Enhancement & Pipeline Routing...")
    if label == "low_light":
        print("--> Routing to WaterNet (Low-Light Enhancement)...")
        selected_file = run_enhancement_model(WaterNet, WATERNET_WEIGHTS, WATERNET_OUT)
    elif label == "turbid":
        print("--> Routing to UWCNN (Turbidity Correction)...")
        selected_file = run_enhancement_model(UWCNN, UWCNN_WEIGHTS, UWCNN_OUT)
    else: # "good"
        print("--> Image quality clear. Passing original image directly to YOLOv8...")
        selected_file = INPUT_IMAGE

    # Copy / Save selected image for final detection
    Image.open(selected_file).convert("RGB").save(FINAL_ENHANCED_OUT)

    print("\nStep 3: Running YOLOv8 Object Detection...")
    run_yolo(FINAL_ENHANCED_OUT)

    print("\n----------------------------------------")
    print("Pipeline Execution Completed Successfully!")
    print("Classifier Result Log:", CLASSIFIER_OUT)
    print("Final Enhanced Image Path:", FINAL_ENHANCED_OUT)
    print("YOLO Detection Output Folder:", YOLO_PROJECT / "pred")
    print("----------------------------------------")