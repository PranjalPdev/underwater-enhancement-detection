import os
import random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# -------------------------
# CUDA Performance Tweak
# -------------------------
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# -------------------------
# Config
# -------------------------
ROOT = r"EUVP/Paired"
SUBSETS = ["underwater_dark", "underwater_imagenet", "underwater_scenes"]

IMG_SIZE = 256
BATCH_SIZE = 16  # Increased batch size for faster GPU throughput
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = r"checkpoints/waternet_best.pth"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("save_model", exist_ok=True)

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -------------------------
# Dataset
# -------------------------
class EUVPPairedDataset(Dataset):
    def __init__(self, root, subsets, split="train", transform=None):
        self.items = []
        self.transform = transform
        for subset in subsets:
            subset_dir = os.path.join(root, subset)
            if split == "train":
                a_dir = os.path.join(subset_dir, "trainA")
                b_dir = os.path.join(subset_dir, "trainB")
                self.items.extend(self._pair_sorted(a_dir, b_dir))
            else:
                val_dir = os.path.join(subset_dir, "validation")
                self.items.extend(self._pair_validation(val_dir))

        if len(self.items) == 0:
            raise RuntimeError("No paired images found. Check EUVP paths.")

    def _image_files(self, d):
        if not os.path.isdir(d):
            return []
        return sorted([
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

    def _pair_sorted(self, a_dir, b_dir):
        a_files = self._image_files(a_dir)
        b_files = self._image_files(b_dir)
        n = min(len(a_files), len(b_files))
        return list(zip(a_files[:n], b_files[:n]))

    def _pair_validation(self, val_dir):
        if not os.path.isdir(val_dir):
            return []
        subdirs = [os.path.join(val_dir, d) for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
        if len(subdirs) >= 2:
            return self._pair_sorted(subdirs[0], subdirs[1])

        files = self._image_files(val_dir)
        if len(files) < 2:
            return []
        half = len(files) // 2
        return list(zip(files[:half], files[half:half * 2]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        a_path, b_path = self.items[idx]
        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("RGB")
        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)
        return a_img, b_img


# -------------------------
# WaterNet Architecture (5 Conv Layers)
# -------------------------
class WaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # net.0
            nn.ReLU(inplace=True),                        # net.1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # net.2
            nn.ReLU(inplace=True),                        # net.3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # net.4
            nn.ReLU(inplace=True),                        # net.5
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # net.6
            nn.ReLU(inplace=True),                        # net.7
            nn.Conv2d(32, 3, kernel_size=3, padding=1),   # net.8
            nn.Sigmoid()                                  # net.9
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Training Loop
# -------------------------
if __name__ == "__main__":
    train_ds = EUVPPairedDataset(ROOT, SUBSETS, split="train", transform=tfm)

    try:
        val_ds = EUVPPairedDataset(ROOT, SUBSETS, split="val", transform=tfm)
        print("Using validation folders.")
    except Exception as e:
        print("Validation folders not usable, falling back to 90/10 split.")
        val_size = max(1, int(0.1 * len(train_ds)))
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])

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

    print("Train pairs:", len(train_ds))
    print("Val pairs:", len(val_ds))

    model = WaterNet().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Updated modern PyTorch AMP GradScaler syntax
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # Updated modern PyTorch AMP autocast syntax
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = model(x)
                    loss = criterion(out, y)

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best checkpoint to {SAVE_PATH}")

    print("Training finished.")