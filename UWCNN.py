import os
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# CUDA Performance Tweak
# -------------------------
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# -------------------------
# Config
# -------------------------
ROOT = r"C:\internnn\UWCNN"
TYPE_FOLDERS = {
    "type2": {
        "gt": "gt_type_2",
        "uw": "underwater_type_2"
    },
    "type3": {
        "gt": "gt_type_3",
        "uw": "underwater_type_3"
    }
}

IMG_SIZE = 256
BATCH_SIZE = 16  # Increased batch size for higher GPU utilization (adjust if OOM)
EPOCHS = 60
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"
BEST_PATH = os.path.join(SAVE_DIR, "uwcnn_best.pth")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("save_model", exist_ok=True)

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -------------------------
# Dataset
# -------------------------
class UWCNNDataset(Dataset):
    def __init__(self, root, type_folders, transform=None):
        self.items = []
        self.transform = transform

        for key, folder_map in type_folders.items():
            gt_dir = os.path.join(root, key, folder_map["gt"])
            uw_dir = os.path.join(root, key, folder_map["uw"])

            if not os.path.isdir(gt_dir) or not os.path.isdir(uw_dir):
                print(f"Skipping missing folder pair: {gt_dir}, {uw_dir}")
                continue

            gt_files = sorted([
                os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ])
            uw_files = sorted([
                os.path.join(uw_dir, f) for f in os.listdir(uw_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ])

            n = min(len(gt_files), len(uw_files))
            self.items.extend(list(zip(uw_files[:n], gt_files[:n])))

        if len(self.items) == 0:
            raise RuntimeError("No paired UWCNN images found. Check your folder paths.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        uw_path, gt_path = self.items[idx]
        uw = Image.open(uw_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        if self.transform:
            uw = self.transform(uw)
            gt = self.transform(gt)

        return uw, gt


# -------------------------
# Official UWCNN Architecture
# (Lightweight Dense Network with Residual Connection)
# -------------------------
class UWCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
        
        # Dense feature concatenation layers
        self.conv4 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(80, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(96, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv2d(112, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(128, 16, 3, padding=1), nn.ReLU(inplace=True))
        
        # Final output conv
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
        
        # Residual addition prevents flat gray image outputs
        return torch.clamp(x + out, 0.0, 1.0)


# -------------------------
# Main Training Loop
# -------------------------
if __name__ == "__main__":
    dataset = UWCNNDataset(ROOT, TYPE_FOLDERS, transform=tfm)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise RuntimeError("Not enough data for train/val split.")

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Multithreaded Data Loading
    num_workers = 4 if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    print("Total pairs:", len(dataset))
    print("Train pairs:", len(train_ds))
    print("Val pairs:", len(val_ds))

    model = UWCNN().to(DEVICE)


    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Automatic Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    
    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for uw, gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            uw = uw.to(DEVICE, non_blocking=True)
            gt = gt.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                out = model(uw)
                loss = criterion(out, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * uw.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for uw, gt in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                uw = uw.to(DEVICE, non_blocking=True)
                gt = gt.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = model(uw)
                    loss = criterion(out, gt)

                val_loss += loss.item() * uw.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), BEST_PATH)
                print(f"Saved best checkpoint to {BEST_PATH}")

    print("Training finished.")