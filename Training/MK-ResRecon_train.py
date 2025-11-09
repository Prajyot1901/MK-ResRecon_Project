# ===================== train_attention_unet.py =====================
import os
import re
import random
import csv
import numpy as np
from math import log10
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# ===================== CONFIG =====================
slices_dir = r"Slices/axial"
image_size = (256, 256)
batch_size = 8
num_epochs = 20
learning_rate = 1e-4
device = "cuda"
test_prefixes_count =80
checkpoint_dir = "Best_model_8"       # Use different checkpoint dir for  4 and 2 gap
os.makedirs(checkpoint_dir, exist_ok=True)
log_csv = "L1_loss_8.csv"                              # Use different filename for  4 and 2 gap
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)

# ===================== DATASET =====================
class SliceDataset(Dataset):
    def __init__(self, slices_dir, selected_prefixes):
        self.samples = []
        files = sorted([f for f in os.listdir(slices_dir) if f.endswith(".png")])
        groups = {}
        for f in files:
            prefix = f[:6]
            groups.setdefault(prefix, []).append(f)
        groups = {k: groups[k] for k in selected_prefixes if k in groups}

        gap = 8                             # 2/4 slice gap
        mid = gap // 2

        for prefix, group in groups.items():
            group.sort(key=lambda x: int(re.findall(r"_(\d+)\.png", x)[0]))
            slice_nums = [int(re.findall(r"_(\d+)\.png", g)[0]) for g in group]
            for i in range(len(slice_nums) - gap):
                s1 = os.path.join(slices_dir, group[i])
                s2 = os.path.join(slices_dir, group[i + gap])
                smid = os.path.join(slices_dir, group[i + mid])
                self.samples.append((s1, s2, smid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s1_path, s2_path, s_mid_path = self.samples[idx]

        def load_img(path):
            img = Image.open(path).convert("L").resize(image_size)
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.tensor(arr).unsqueeze(0)

        s1 = load_img(s1_path)
        s2 = load_img(s2_path)
        s_mid = load_img(s_mid_path)
        inp = torch.cat([s1, s2], dim=0)
        return inp, s_mid

# ===================== MODULES =====================
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ===================== Attention Residual UNet (Plain) =====================
class AttentionResUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, filters=[32, 64, 128, 256, 512, 1024]):
        super().__init__()
        # Encoder (no dilation)
        self.enc1 = ResidualConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = ResidualConvBlock(filters[3], filters[4])
        self.pool5 = nn.MaxPool2d(2)
        self.bottleneck = ResidualConvBlock(filters[4], filters[5])

        # Decoder
        self.up5 = nn.ConvTranspose2d(filters[5], filters[4], kernel_size=2, stride=2)
        self.att5 = AttentionGate(filters[4], filters[4], filters[4] // 2)
        self.dec5 = ResidualConvBlock(filters[5], filters[4])

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(filters[3], filters[3], filters[3] // 2)
        self.dec4 = ResidualConvBlock(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(filters[2], filters[2], filters[2] // 2)
        self.dec3 = ResidualConvBlock(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(filters[1], filters[1], filters[1] // 2)
        self.dec2 = ResidualConvBlock(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(filters[0], filters[0], filters[0] // 2)
        self.dec1 = ResidualConvBlock(filters[1], filters[0])

        self.final_conv = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))
        b = self.bottleneck(self.pool5(e5))

        d5 = self.dec5(torch.cat([self.up5(b), self.att5(e5, self.up5(b))], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), self.att4(e4, self.up4(d5))], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(e3, self.up3(d4))], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(e2, self.up2(d3))], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(e1, self.up1(d2))], dim=1))
        return self.final_conv(d1)

# ===================== SSIM =====================
def gaussian_window(window_size, sigma, channel, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = (window_1d @ window_1d.t()).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()

def ssim_pytorch(img1, img2, window_size=11, window_sigma=1.5, data_range=1.0, K=(0.01, 0.03), reduction='mean'):
    B, C, H, W = img1.shape
    device = img1.device
    window = gaussian_window(window_size, window_sigma, C, device)
    mu1 = nn.functional.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = nn.functional.conv2d(img2, window, padding=window_size // 2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2
    C1, C2 = (K[0] * data_range)**2, (K[1] * data_range)**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'map':
        return ssim_map.mean(dim=1, keepdim=True)
    else:
        return ssim_map

# ===================== LOSSES =====================
def mae_loss(pred, target): return torch.mean(torch.abs(pred - target))
def l1_loss(pred, target): return nn.functional.l1_loss(pred, target)
def psnr_loss(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return torch.tensor(0.0, device=pred.device) if mse == 0 else -10 * torch.log10(1.0 / mse)
def ssim_loss(pred, target):
    return 1.0 - ssim_pytorch(pred, target, reduction='mean', data_range=1.0)


def filtered_l1_loss(pred, target):
    # Define filters
    filters = torch.tensor([
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],  # horizontal

        [[[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]],  # vertical

        [[[0, 1, 0],
          [1, -4, 1],
          [0, 1, 0]]],  # laplacian

        [[[1, -1, 1],
          [-1, 1, -1],
          [1, -1, 1]]],  # checkerboard

        [[[0, 1, 2],
          [-1, 0, 1],
          [-2, -1, 0]]],  # diagonal 1

        [[[2, 1, 0],
          [1, 0, -1],
          [0, -1, -2]]]   # diagonal 2
    ], dtype=pred.dtype, device=pred.device)

    # Assign custom weights to each filter
    weights = torch.tensor([0.1, 0.1, 2.0, 0, 0.5, 0.5], dtype=pred.dtype, device=pred.device)
    weights = weights / weights.sum()  # normalize to sum=1 for balance

    # Apply filters
    pred_f = nn.functional.conv2d(pred, filters, padding=1)
    targ_f = nn.functional.conv2d(target, filters, padding=1)

    # Compute weighted L1 loss for each filtered response
    per_filter_loss = torch.mean(torch.abs(pred_f - targ_f), dim=[1, 2, 3])  # (batch,)
    weighted_loss = (weights.view(-1, 1) * torch.mean(torch.abs(pred_f - targ_f), dim=[0, 2, 3])).sum()

    return weighted_loss



# ===================== MAIN =====================
if __name__ == "__main__":
    # Load prefixes from txt file
    txt_file = r"D:\Project_Yale\Slices_2\POST.txt"  # your text file with selected prefixes
    with open(txt_file, "r") as f:
        prefixes = [line.strip() for line in f if line.strip()]
    
    print(f"✅ Loaded {len(prefixes)} prefixes from {txt_file}")

    # Split into train/test
    test_ratio = test_prefixes_count / max(1, len(prefixes))
    train_prefixes, test_prefixes = train_test_split(prefixes, test_size=test_ratio, random_state=seed)
    print(f"✅ Using {len(train_prefixes)} training and {len(test_prefixes)} testing prefixes.")

    # Create datasets
    train_ds = SliceDataset(slices_dir, train_prefixes)
    test_ds = SliceDataset(slices_dir, test_prefixes)
    print(f"📁 Training samples: {len(train_ds)} | Testing samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = AttentionResUNet(in_channels=2, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    writer = SummaryWriter()

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_mae", "val_psnr", "val_ssim", "lr"])

    best_val_psnr = -1e9
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = (
                0 * l1_loss(outputs, targets)  + # base pixel similarity
                1.0 * filtered_l1_loss(outputs, targets)  # emphasize edges/textures
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)

        model.eval()
        mae_sum = psnr_sum = ssim_sum = count = 0.0
        with torch.no_grad():
            first_val_batch = None
            for inputs, targets in test_loader:
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
                if first_val_batch is None:
                    first_val_batch = (inputs.cpu(), outputs.cpu(), targets.cpu())

        val_mae = mae_sum / count if count else float("nan")
        val_psnr = psnr_sum / count if count else float("nan")
        val_ssim = ssim_sum / count if count else float("nan")
        scheduler.step(val_psnr)

        if best_val_psnr < val_psnr:
            best_val_psnr = val_psnr
            ckpt_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr
            }, ckpt_path)
            best_msg = " (best -> saved)"
        else:
            best_msg = ""

        current_lr = optimizer.param_groups[0]['lr']
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_train_loss:.6f}", f"{val_mae:.6f}",
                                    f"{val_psnr:.4f}", f"{val_ssim:.6f}", f"{current_lr:.6e}"])

        writer.add_scalar("Val/MAE", val_mae, epoch)
        writer.add_scalar("Val/PSNR", val_psnr, epoch)
        writer.add_scalar("Val/SSIM", val_ssim, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        if first_val_batch is not None:
            inp_cpu, out_cpu, tgt_cpu = first_val_batch
            nshow = min(4, inp_cpu.shape[0])
            img_list = []
            for i in range(nshow):
                img_list.extend([
                    inp_cpu[i, 0:1],
                    inp_cpu[i, 1:2],
                    out_cpu[i, 0:1],
                    tgt_cpu[i, 0:1],
                ])
            grid = make_grid(img_list, nrow=4, normalize=True)
            save_image(grid, os.path.join(checkpoint_dir, f"epoch_{epoch}_comparison.png"))
            writer.add_image("Comparison", grid, epoch)

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | "
              f"Val MAE: {val_mae:.6f} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.6f} | LR: {current_lr:.6e}{best_msg}")

    writer.close()
