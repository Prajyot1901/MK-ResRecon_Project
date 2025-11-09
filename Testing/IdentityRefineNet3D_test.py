import os
import re
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# ===================== CONFIG =====================
gen_dir = r"Path/to/generated/images"
gt_dir = r"Path/to/ground_truth/images"
checkpoint_path = r"Path/to/downloaded/model/weights"
device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = r"Path/to/output/directory"
os.makedirs(save_dir, exist_ok=True)


# ===================== HELPERS (Copied Exactly from Training) =====================
def get_groupnorm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1:
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
        groups -= 1
    return nn.GroupNorm(1, channels)


class SliceAttention3D(nn.Module):
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


class IdentityRefineNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=2, depth=256):
        super().__init__()
        self.slice_attention = SliceAttention3D(depth)

        self.conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.gn = get_groupnorm(base_ch)
        self.act = nn.ReLU(inplace=True)

        self.res_conv = nn.Conv3d(base_ch, base_ch, 3, padding=1)
        self.res_gn = get_groupnorm(base_ch)

        self.out_conv = nn.Conv3d(base_ch, out_ch, 1)
        self.identity = nn.Identity()

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


# ===================== METRICS (same functions) =====================
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return (10 * torch.log10(1.0 / mse)).item()

def ssim3d(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = torch.mean(pred)
    mu_y = torch.mean(target)
    sigma_x = torch.var(pred)
    sigma_y = torch.var(target)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return (num / den).item()


# ===================== INFERENCE =====================
def infer_volume(gen_path, gt_path, model):
    gen_vol = nib.load(gen_path).get_fdata().astype(np.float32)
    gt_vol = nib.load(gt_path).get_fdata().astype(np.float32)

    if np.max(gen_vol) > 0: gen_vol = gen_vol / np.max(gen_vol)
    if np.max(gt_vol) > 0: gt_vol = gt_vol / np.max(gt_vol)

    gen_tensor = torch.from_numpy(gen_vol).unsqueeze(0).unsqueeze(0).to(device)
    gt_tensor = torch.from_numpy(gt_vol).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(gen_tensor)

    return gen_tensor, gt_tensor, out


# ===================== MAIN =====================
if __name__ == "__main__":
    print("Loading model...")

    model = IdentityRefineNet3D().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Loaded checkpoint from: {checkpoint_path}")

    gen_files = {re.match(r"(\w{6})", f).group(1): f
                 for f in os.listdir(gen_dir) if f.endswith(".nii.gz")}
    gt_files = {re.match(r"(\w{6})", f).group(1): f
                for f in os.listdir(gt_dir) if f.endswith(".nii.gz")}

    common = sorted(set(gen_files) & set(gt_files))
    print(f"Found {len(common)} matching 3D volumes.\n")

    total_psnr = 0
    total_ssim = 0
    results = []
    for p in common:
        gen_path = os.path.join(gen_dir, gen_files[p])
        gt_path = os.path.join(gt_dir, gt_files[p])
        
        
        gen_img = nib.load(gen_path)
        affine = gen_img.affine
        header = gen_img.header
        
        
        gen_t, gt_t, pred_t = infer_volume(gen_path, gt_path, model)

        p_psnr = psnr(pred_t, gt_t)
        p_ssim = ssim3d(pred_t, gt_t)

        total_psnr += p_psnr
        total_ssim += p_ssim

        print(f"{p} → PSNR: {p_psnr:.4f}, SSIM: {p_ssim:.4f}")
        results.append({
            "Prefix": p,
            "PSNR": p_psnr,
            "SSIM": p_ssim
        })
        
        pred_np = pred_t.squeeze().cpu().numpy()  # (D, H, W)
        save_path = os.path.join(save_dir, f"{p}_refined.nii.gz")
        nib.save(nib.Nifti1Image(pred_np, affine, header), save_path)
        print(f"Saved refined volume → {save_path}")
    csv_path = os.path.join(save_dir, "Name_of_output_file.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"📊 Saved metrics CSV → {csv_path}")
    avg_psnr = total_psnr / len(common)
    avg_ssim = total_ssim / len(common)

    print("\n============================")
    print(f"AVERAGE PSNR: {avg_psnr:.4f}")
    print(f"AVERAGE SSIM: {avg_ssim:.4f}")
    print("============================")
