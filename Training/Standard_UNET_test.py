# ===================== test_simple_unet.py =====================
import os
import re
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from Simple_UNET import UNet, ssim_pytorch, SliceDataset  # reuse your definitions

# ===================== CONFIG =====================
slices_dir = r"Slices/axial"
txt_file = r"D:\Project_Yale\Slices_2\POST.txt"  # same prefix file
checkpoint_path = None  # <- set automatically to best checkpoint below

image_size = (256, 256)
batch_size = 8
device = "cuda"
test_prefixes_count = 80
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)

# ===================== FIND LATEST/BEST CHECKPOINT =====================

checkpoint_path = r"D:\Project_Yale\simple_unet_checkpoints_2\best_model_epoch_10_psnr_31.30.pth"
print(f"✅ Loaded checkpoint: {checkpoint_path}")

# ===================== LOAD TEST DATASET =====================
with open(txt_file, "r") as f:
    prefixes = [line.strip() for line in f if line.strip()]

test_ratio = test_prefixes_count / max(1, len(prefixes))
_, test_prefixes = train_test_split(prefixes, test_size=test_ratio, random_state=seed)

test_ds = SliceDataset(slices_dir, test_prefixes)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
print(f"📁 Testing samples: {len(test_ds)}")

# ===================== LOAD MODEL =====================
model = UNet(in_channels=1, out_channels=1).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ===================== TEST LOOP =====================
mae_sum = psnr_sum = ssim_sum = count = 0.0
save_dir = "test_results"
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        abs_err = torch.mean(torch.abs(outputs - targets), dim=[1, 2, 3])
        mae_sum += float(abs_err.sum().item())

        mse_batch = torch.mean((outputs - targets) ** 2, dim=[1, 2, 3])
        psnr_batch = 10.0 * torch.log10(1.0 / (mse_batch + 1e-12))
        psnr_sum += float(psnr_batch.sum().item())

        for i in range(outputs.shape[0]):
            ssim_val = ssim_pytorch(outputs[i:i+1], targets[i:i+1], reduction='mean', data_range=1.0)
            ssim_sum += float(ssim_val.item())
            count += 1

        # Optionally save some example predictions
        if batch_idx < 5:
            for i in range(min(outputs.shape[0], 4)):
                save_image(outputs[i], os.path.join(save_dir, f"pred_{batch_idx:03d}_{i}.png"))
                save_image(targets[i], os.path.join(save_dir, f"gt_{batch_idx:03d}_{i}.png"))

# ===================== RESULTS =====================
val_mae = mae_sum / count if count else float("nan")
val_psnr = psnr_sum / count if count else float("nan")
val_ssim = ssim_sum / count if count else float("nan")

print("\n===== TEST RESULTS =====")
print(f"MAE :  {val_mae:.6f}")
print(f"PSNR:  {val_psnr:.4f}")
print(f"SSIM:  {val_ssim:.6f}")
print("=========================")
print(f"✅ Sample predictions saved to: {save_dir}")
