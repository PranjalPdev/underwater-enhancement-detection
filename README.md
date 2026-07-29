# Underwater Image Enhancement & Object Detection Pipeline

An end-to-end deep learning pipeline that classifies underwater image quality, dynamically routes images to targeted enhancement models (WaterNet or UWCNN), and performs object detection using YOLOv8.

## 📌 Architecture Pipeline
1. **Quality Classifier:** TensorFlow EfficientNet (`low_light`, `turbid`, or `good`).
2. **Enhancement Routing:**
   - `low_light` ➔ **WaterNet** (PyTorch)
   - `turbid` ➔ **UWCNN** (PyTorch)
   - `good` ➔ Direct to YOLOv8
3. **Object Detection:** Custom-trained **YOLOv8** on URPC2020 dataset.

## 📂 Model Weights
Download the trained model checkpoints from Google Drive and place them inside the `model/` directory:
👉 **[Click Here to Download Model Weights](https://drive.google.com/drive/folders/1Y2IkD4FCaJkq0KuUHBH9s0gs2MV8FN9D?usp=sharing)**

Place files in:
- `model/efficientnet_classifier.keras`
- `model/waternet_best.pth`
- `model/uwcnn_best.pth`
- `model/best.pt`

## 🚀 How to Run

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt