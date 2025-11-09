
import os
import re
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from math import log10

slices_dir = r"Path/to/slices"       # Change this path
image_size = (256, 256)
batch_size = 16  # larger batch for faster eval
device = "cuda" if torch.cuda.is_available() else "cpu"
test_prefixes_count = 80
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)

# ================================================================
# DATASET (same as in training)
# ================================================================
class SliceDataset(Dataset):
    def __init__(self, slices_dir, selected_prefixes):
        self.samples = []
        files = sorted([f for f in os.listdir(slices_dir) if f.endswith(".png")])
        groups = {}
        for f in files:
            prefix = f[:6]
            groups.setdefault(prefix, []).append(f)
        groups = {k: groups[k] for k in selected_prefixes if k in groups}

        gap = 8                                                                 # Can be changed to 4 or 2
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

# ================================================================
# MODEL (exact same architecture)
# ================================================================
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
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
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class AttentionResUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, filters=[32, 64, 128, 256, 512, 1024]):
        super().__init__()
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

        self.up5 = nn.ConvTranspose2d(filters[5], filters[4], 2, 2)
        self.att5 = AttentionGate(filters[4], filters[4], filters[4] // 2)
        self.dec5 = ResidualConvBlock(filters[5], filters[4])

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, 2)
        self.att4 = AttentionGate(filters[3], filters[3], filters[3] // 2)
        self.dec4 = ResidualConvBlock(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, 2)
        self.att3 = AttentionGate(filters[2], filters[2], filters[2] // 2)
        self.dec3 = ResidualConvBlock(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, 2)
        self.att2 = AttentionGate(filters[1], filters[1], filters[1] // 2)
        self.dec2 = ResidualConvBlock(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, 2)
        self.att1 = AttentionGate(filters[0], filters[0], filters[0] // 2)
        self.dec1 = ResidualConvBlock(filters[1], filters[0])

        self.final_conv = nn.Sequential(nn.Conv2d(filters[0], out_channels, 1), nn.Sigmoid())

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

# ================================================================
# METRICS
# ================================================================
def ssim_pytorch(img1, img2, window_size=11, window_sigma=1.5, data_range=1.0):
    B, C, H, W = img1.shape
    device = img1.device
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * window_sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = (window_1d @ window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(C, 1, window_size, window_size)
    mu1 = nn.functional.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = nn.functional.conv2d(img2, window, padding=window_size // 2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2
    C1, C2 = (0.01 * data_range)**2, (0.03 * data_range)**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    txt_file = r"Path/to/txt/file"
    ckpt_path = r"Path/to/downloaded/weights"         # update this path!

    with open(txt_file, "r") as f:
        prefixes = [line.strip() for line in f if line.strip()]
    test_ratio = test_prefixes_count / max(1, len(prefixes))
    _, test_prefixes = train_test_split(prefixes, test_size=test_ratio, random_state=seed)

    test_ds = SliceDataset(slices_dir, test_prefixes)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"📁 Test samples: {len(test_ds)} | Batches: {len(test_loader)}")

    model = AttentionResUNet(in_channels=2, out_channels=1).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mae_sum = psnr_sum = ssim_sum = count = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)

            abs_err = torch.abs(outputs - targets)
            mae_sum += abs_err.mean().item() * inputs.size(0)

            mse = (abs_err ** 2).mean(dim=[1, 2, 3])
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
            psnr_sum += psnr.sum().item()

            for i in range(outputs.size(0)):
                ssim_sum += ssim_pytorch(outputs[i:i+1], targets[i:i+1]).item()
            count += inputs.size(0)

    print("========== EVALUATION RESULTS ==========")
    print(f"MAE  : {mae_sum / count:.6f}")
    print(f"PSNR : {psnr_sum / count:.4f} dB")
    print(f"SSIM : {ssim_sum / count:.6f}")
    print("========================================")
