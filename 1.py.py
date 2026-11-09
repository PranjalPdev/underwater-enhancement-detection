from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

from zipfile import ZipFile

zip_path = Path(r'C:\Users\pppan\OneDrive\Desktop\internnn\EUVP.zip')
print("Zip exists:", zip_path.exists())

with ZipFile(zip_path, 'r') as z:
    names = z.namelist()
    for n in names[:50]:
        print(n)

img_exts = {'.jpg', '.JPEG', '.png', '.bmp', '.tif', '.tiff'}
all_imgs = [p for p in zip_path.rglob('*') if p.suffix.lower() in img_exts]
print('Total images found:', len(all_imgs))

sample_paths = random.sample(all_imgs, min(6, len(all_imgs)))
fig, axes = plt.subplots(len(sample_paths), 1, figsize=(8, 4 * len(sample_paths)))

if len(sample_paths) == 1:
    axes = [axes]

for ax, p in zip(axes, sample_paths):
    try:
        img = Image.open(p).convert('RGB')
        ax.imshow(img)
        ax.set_title(str(p.relative_to(zip_path)))
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Failed to open:\n{p.name}\n{e}', ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
plt.show()