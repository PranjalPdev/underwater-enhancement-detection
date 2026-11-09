import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO

# ==========================================
# Paths & Config
# ==========================================
BASE = Path(r"C:\final")
INPUT_IMAGE = BASE / "input" / "demo.jpg"
OUTPUT_DIR = BASE / "output"
MODELS_DIR = BASE / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TensorFlow / Keras Classifier Model & Indices
CLASSIFIER_WEIGHTS = MODELS_DIR / "efficientnet_classifier.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# PyTorch Enhancement Models & YOLO
WATERNET_WEIGHTS = MODELS_DIR / "waternet.pth"
UWCNN_WEIGHTS = MODELS_DIR / "uwcnn.pth"
YOLO_WEIGHTS = MODELS_DIR / "best.pt"

CLASSIFIER_OUT = OUTPUT_DIR / "classifier_result.txt"
WATERNET_OUT = OUTPUT_DIR / "waternet_enhanced.jpg"
UWCNN_OUT = OUTPUT_DIR / "uwcnn_enhanced.jpg"
FINAL_ENHANCED_OUT = OUTPUT_DIR / "selected_enhanced.jpg"
YOLO_PROJECT = OUTPUT_DIR / "yolo_results"

img_size = 256

# ==========================================
# Helper Functions (PyTorch Encoders)
# ==========================================
def load_image_tensor(img_path, size=256):
    img = Image.open(img_path).convert("RGB").resize((size, size))
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32).view(size, size, 3) / 255.0
    return arr.permute(2, 0, 1).unsqueeze(0)

def save_tensor_image(tensor, out_path):
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(out_path)

# ==========================================
# Enhancement Models (PyTorch)
# ==========================================
class WaterNetNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 3, 3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class UWCNNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 3, 3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# Execution Functions
# ==========================================
def run_classifier():
    """Runs prediction using the TensorFlow Keras EfficientNet model."""
    if not CLASSIFIER_WEIGHTS.exists():
        print(f"Warning: Classifier weights not found at {CLASSIFIER_WEIGHTS}. Defaulting to 'good'.")
        return "good"

    # 1. Load Keras Model
    model = tf.keras.models.load_model(CLASSIFIER_WEIGHTS)
    
    # Get model's expected target image size (fallback to 224x224 or img_size if not dynamic)
    input_shape = model.input_shape[1:3]
    target_size = input_shape if None not in input_shape else (img_size, img_size)

    # 2. Preprocess input image for TensorFlow
    img = Image.open(INPUT_IMAGE).convert("RGB").resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, H, W, 3)

    # 3. Perform Inference
    predictions = model.predict(img_batch, verbose=0)
    idx = int(np.argmax(predictions, axis=1)[0])

    # 4. Map Class Index to Label
    label_map = {}
    if CLASS_INDICES_PATH.exists():
        try:
            with open(CLASS_INDICES_PATH, "r") as f:
                class_indices = json.load(f)
            # Reverse mapping from index -> label name
            label_map = {int(v): k for k, v in class_indices.items()}
        except Exception as e:
            print(f"Could not parse class_indices.json: {e}")

    # Fallback dictionary mapping if JSON file is absent or incomplete
    if not label_map:
        label_map = {0: "good", 1: "low_light", 2: "turbid"}

    label = label_map.get(idx, "good")
    
    # Save output summary text
    CLASSIFIER_OUT.write_text(f"image={INPUT_IMAGE.name}\nlabel={label}\nclass_index={idx}\n")
    return label

def run_waternet(out_path):
    x = load_image_tensor(INPUT_IMAGE, img_size)
    model = WaterNetNet()
    if WATERNET_WEIGHTS.exists():
        model.load_state_dict(torch.load(WATERNET_WEIGHTS, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        y = model(x)
    save_tensor_image(y, out_path)
    return out_path

def run_uwcnn(out_path):
    x = load_image_tensor(INPUT_IMAGE, img_size)
    model = UWCNNNet()
    if UWCNN_WEIGHTS.exists():
        model.load_state_dict(torch.load(UWCNN_WEIGHTS, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        y = model(x)
    save_tensor_image(y, out_path)
    return out_path

def run_yolo(source_img):
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
# Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Missing input image: {INPUT_IMAGE}")

    # 1. Run TF Classifier
    label = run_classifier()
    print(f"Detected Label: {label}")

    # 2. Image Selection & Enhancement Routing
    if label == "low_light":
        selected_file = run_waternet(WATERNET_OUT)
    elif label == "turbid":
        selected_file = run_uwcnn(UWCNN_OUT)
    else:
        selected_file = INPUT_IMAGE

    # 3. Save Selected / Enhanced Image
    Image.open(selected_file).convert("RGB").save(FINAL_ENHANCED_OUT)

    # 4. Run Object Detection via YOLO
    run_yolo(FINAL_ENHANCED_OUT)

    # Summary Prints
    print("----------------------------------------")
    print("Classifier Results:", CLASSIFIER_OUT)
    print("Final Enhanced Image:", FINAL_ENHANCED_OUT)
    print("YOLO Output Folder:", YOLO_PROJECT)
    print("----------------------------------------")