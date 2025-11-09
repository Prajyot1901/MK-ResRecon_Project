import os
import re
import random
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===================== CONFIG =====================
gen_dir = r"D:\Project_Yale\Generated_3D"
gt_dir = r"D:\Project_Yale\Resized_3D"
train_txt = r"D:\Project_Yale\Slices_2\train_prefixes.txt"
test_txt = r"D:\Project_Yale\Slices_2\test_prefixes.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_epochs = 50
lr = 1e-4
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# ===================== DATASET =====================
class VolumeDataset(Dataset):
    def __init__(self, gen_dir, gt_dir, prefixes):
        self.samples = []
        gen_files = {}
        gt_files = {}
        for f in os.listdir(gen_dir):
            if f.endswith(".nii.gz"):
                m = re.match(r"(\w{6})", f)
                if m:
                    gen_files[m.group(1)] = os.path.join(gen_dir, f)
        for f in os.listdir(gt_dir):
            if f.endswith(".nii.gz"):
                m = re.match(r"(\w{6})", f)
                if m:
                    gt_files[m.group(1)] = os.path.join(gt_dir, f)
        for p in prefixes:
            if p in gen_files and p in gt_files:
                self.samples.append((gen_files[p], gt_files[p]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gen_path, gt_path = self.samples[idx]
        gen_vol = nib.load(gen_path).get_fdata().astype(np.float32)
        gt_vol = nib.load(gt_path).get_fdata().astype(np.float32)

        # Avoid division by zero
        if np.max(gen_vol) > 0:
            gen_vol = gen_vol / np.max(gen_vol)
        if np.max(gt_vol) > 0:
            gt_vol = gt_vol / np.max(gt_vol)

        gen_vol = torch.from_numpy(gen_vol).unsqueeze(0)  # (1, D, H, W)
        gt_vol = torch.from_numpy(gt_vol).unsqueeze(0)
        return gen_vol, gt_vol

# ===================== HELPERS =====================
def get_groupnorm(channels, max_groups=8):
    # choose groups <= max_groups and divides channels
    groups = min(max_groups, channels)
    while groups > 1:
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
        groups -= 1
    return nn.GroupNorm(1, channels)  # fallback to LayerNorm-like behavior

class SliceAttention3D(nn.Module):
    """Lightweight slice attention along depth."""
    def __init__(self, depth=256):
        super().__init__()
        self.depth = depth
        self.attention = nn.Parameter(torch.ones(depth))
        mask = torch.zeros(depth)
        mask[::8] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        weights = torch.sigmoid(self.attention) * self.mask + (1 - self.mask)
        return x * weights.view(1, 1, self.depth, 1, 1)
# ===================== MINIMAL 3D REFINEMENT NET =====================
class IdentityRefineNet3D(nn.Module):
    """
    Minimal 3D refinement network for 256^3 volumes.
    - Very few channels (2-4)
    - Only 1 residual conv layer
    - Slice attention kept
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=2, depth=256):
        super().__init__()
        self.slice_attention = SliceAttention3D(depth)

        # Single conv layer
        self.conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.gn = get_groupnorm(base_ch)
        self.act = nn.ReLU(inplace=True)

        # Residual refinement (just 1 conv)
        self.res_conv = nn.Conv3d(base_ch, base_ch, 3, padding=1)
        self.res_gn = get_groupnorm(base_ch)

        # Output
        self.out_conv = nn.Conv3d(base_ch, out_ch, 1)
        self.identity = nn.Identity()

        # weight init
        for m in [self.conv, self.res_conv, self.out_conv]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        if hasattr(self.gn, "weight") and self.gn.weight is not None: nn.init.ones_(self.gn.weight)
        if hasattr(self.gn, "bias") and self.gn.bias is not None: nn.init.zeros_(self.gn.bias)

    def forward(self, x):
        x_att = self.slice_attention(x)
        h = self.conv(x_att)
        h = self.gn(h)
        h = self.act(h)

        h_res = self.res_conv(h)
        h_res = self.res_gn(h_res)
        h_res = self.act(h + h_res)

        out = torch.sigmoid(self.out_conv(h_res))
        refined = torch.clamp(x + 0.005 * out, 0.0, 1.0)
        return refined


# ===================== FILTERED L1 LOSS (unchanged) =====================
def filtered_l1_loss_3d(pred, target):
    """
    Enhanced 3D filtered L1 loss with an expanded filter bank.
    Captures gradients, edges, diagonals, corners, and local texture patterns.
    """
    assert pred.ndim == 5 and target.ndim == 5, "Inputs must be 5D (B, C, D, H, W)"
    assert pred.shape[1] == 1, "Expected single-channel input"

    device = pred.device
    dtype = pred.dtype

    filters = torch.tensor([
        # Sobel X
        [[[[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]],
           [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]],
           [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]
        ]]],
        # Sobel Y
        [[[[
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]],
           [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]],
           [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]
        ]]],
        # Sobel Z
        [[[[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]],
           [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
           [[-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1]]
        ]]],
        # Laplacian
        [[[[
            [0, 1, 0],
            [1, -6, 1],
            [0, 1, 0]],
           [[1, -6, 1],
            [-6, 24, -6],
            [1, -6, 1]],
           [[0, 1, 0],
            [1, -6, 1],
            [0, 1, 0]]
        ]]],
        # Diagonal 1
        [[[[
            [0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]],
           [[0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]],
           [[0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]]
        ]]],
        # Diagonal 2
        [[[[
            [2, 1, 0],
            [1, 0, -1],
            [0, -1, -2]],
           [[2, 1, 0],
            [1, 0, -1],
            [0, -1, -2]],
           [[2, 1, 0],
            [1, 0, -1],
            [0, -1, -2]]
        ]]],
        # Gaussian smoothing
        [[[[
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]],
           [[2, 4, 2],
            [4, 8, 4],
            [2, 4, 2]],
           [[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]
        ]]],
        # High-pass edge
        [[[[
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]],
           [[-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]],
           [[-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]]
        ]]],
        # Laplacian of Gaussian (LoG)
        [[[[
            [0, 0, -1],
            [0, -1, -2],
            [-1, -2, -1]],
           [[0, -1, -2],
            [-1, 16, -2],
            [-2, -1, 0]],
           [[-1, -2, -1],
            [-2, -1, 0],
            [0, 0, 0]]
        ]]],
        # Cross diagonal
        [[[[
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]],
           [[1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]],
           [[1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]]
        ]]],
        # Checkerboard pattern
        [[[[
            [1, -1, 1],
            [-1, 1, -1],
            [1, -1, 1]],
           [[-1, 1, -1],
            [1, -1, 1],
            [-1, 1, -1]],
           [[1, -1, 1],
            [-1, 1, -1],
            [1, -1, 1]]
        ]]],
    ], dtype=dtype, device=device).squeeze(1)

    # Normalize filters to control magnitude
    filters = filters / (filters.abs().sum(dim=(1, 2, 3, 4), keepdim=True) + 1e-6)

    # Assign weights (edges > diagonals > smooth)
    weights = torch.tensor(
        [1, 1, 1, 2, 0.8, 0.8, 0.5, 1.2, 1.5, 0.8, 0.5],
        dtype=dtype, device=device
    )
    weights = weights / weights.sum()

    # Apply 3D conv (multi-filter bank)
    pred_f = F.conv3d(pred, filters, padding=1)
    targ_f = F.conv3d(target, filters, padding=1)

    # Compute L1 per filter
    diff = torch.abs(pred_f - targ_f)
    per_filter_loss = diff.mean(dim=[0, 2, 3, 4])
    weighted_loss = torch.sum(weights * per_filter_loss)

    return weighted_loss


# ===================== WEIGHTED L1 LOSS =====================
def weighted_l1_loss(pred, target):
    B, C, D, H, W = pred.shape
    device = pred.device
    weights = torch.ones(D, device=device)
    abs_diff = torch.abs(pred - target)
    weighted = abs_diff * weights.view(1, 1, D, 1, 1)
    loss = weighted.mean() / weights.mean()
    return loss

# ===================== METRICS =====================
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)

def ssim3d(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = torch.mean(pred)
    mu_y = torch.mean(target)
    sigma_x = torch.var(pred)
    sigma_y = torch.var(target)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))
    return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

def evaluate_model(model, loader):
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    count = 0
    with torch.no_grad():
        for gen_vol, gt_vol in loader:
            gen_vol, gt_vol = gen_vol.to(device), gt_vol.to(device)
            out = model(gen_vol)
            psnr_sum += psnr(out, gt_vol).item()
            ssim_sum += ssim3d(out, gt_vol).item()
            count += 1
    if count == 0:
        return 0.0, 0.0
    return psnr_sum / count, ssim_sum / count

def evaluate_input_vs_gt(loader):
    psnr_sum, ssim_sum = 0.0, 0.0
    count = 0
    with torch.no_grad():
        for gen_vol, gt_vol in loader:
            gen_vol, gt_vol = gen_vol.to(device), gt_vol.to(device)
            psnr_sum += psnr(gen_vol, gt_vol).item()
            ssim_sum += ssim3d(gen_vol, gt_vol).item()
            count += 1
    if count == 0:
        return 0.0, 0.0
    return psnr_sum / count, ssim_sum / count

# ===================== MAIN =====================
if __name__ == "__main__":
    import datetime
    from pathlib import Path
    import sys

    # ---- Logging setup ----
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_log_{timestamp}.txt"

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)

    # ---- Checkpoint setup ----
    best_psnr = -float("inf")
    checkpoint_path = r"D:\Project_Yale\Refinement_checkpoint_extralight\best_model_8_00025.pt"

    # ---- Load prefixes from text files ----
    def load_prefixes(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    train_prefixes = load_prefixes(train_txt)
    test_prefixes = load_prefixes(test_txt)

    # ---- Detect unknown prefixes ----
    all_prefixes = sorted({re.match(r"(\w{6})", f).group(1)
                           for f in os.listdir(gen_dir) if f.endswith(".nii.gz")})
    known = set(train_prefixes) | set(test_prefixes)
    unknown_prefixes = [p for p in all_prefixes if p not in known]

    print(f"Train: {len(train_prefixes)} | Test: {len(test_prefixes)} | Unknown: {len(unknown_prefixes)}")

    # ---- Create datasets ----
    train_ds = VolumeDataset(gen_dir, gt_dir, train_prefixes)
    test_ds = VolumeDataset(gen_dir, gt_dir, test_prefixes)
    unknown_ds = VolumeDataset(gen_dir, gt_dir, unknown_prefixes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1,num_workers=4, shuffle=False)
    unknown_loader = DataLoader(unknown_ds, batch_size=1,num_workers=4, shuffle=False)

    # ---- Model setup ----
    model = IdentityRefineNet3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = weighted_l1_loss
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    try:
        model_psnr, model_ssim = evaluate_model(model, test_loader)
        print(f"Initial Model Output→GT PSNR: {model_psnr:.4f}, SSIM: {model_ssim:.4f}\n")
    except Exception as e:
        print(f"Initial evaluation skipped due to: {e}")

    # ---- Training loop with checkpointing ----
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for gen_vol, gt_vol in train_loader:
            gen_vol, gt_vol = gen_vol.to(device), gt_vol.to(device)
            optimizer.zero_grad()
            out = model(gen_vol)
            filtered_weight = 1.0 - (epoch - 1) / max(1, (num_epochs - 1))
            filtered_weight = 0.2 + 0.8 * filtered_weight
            l1_weight = 0.25
            loss = l1_weight * criterion(out, gt_vol) + filtered_weight * filtered_l1_loss_3d(out,gt_vol)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        psnr_val, ssim_val = evaluate_model(model, test_loader)
        psnr_unknown, ssim_unknown = evaluate_model(model, unknown_loader)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"| Train Loss: {avg_loss:.6f} "
              f"| FilteredWeight: {filtered_weight:.4f} "
              f"| Test PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f} "
              f"| Unknown PSNR: {psnr_unknown:.4f}, SSIM: {ssim_unknown:.4f}")

        # ---- Save best model ----
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': psnr_val,
                'ssim': ssim_val,
            }, checkpoint_path)
            print(f"Saved new best model at epoch {epoch} with PSNR {psnr_val:.4f}")

    final_psnr, final_ssim = evaluate_model(model, test_loader)
    print(f"\nFinal Model Output→GT PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}")
    print(f"\nBest model saved at: {checkpoint_path}")
    log_file.close()
