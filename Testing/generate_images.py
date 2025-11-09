import os
import re
import shutil
import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image

# Import your AttentionResUNet definition
from my_loss import AttentionResUNet  # ensure this file is in the same directory

# ========== CONFIG ==========
input_root = r"Path/to/input/directory"     # original slices
output_root = r"Path/to/output/directory"   # generated + copied slices
checkpoint_path = r"Path/to/weights"
image_size = (256, 256)
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(output_root, exist_ok=True)

# ========== MODEL LOAD ==========
print("🔹 Loading model from:", checkpoint_path)
model = AttentionResUNet(in_channels=2, out_channels=1).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ Model loaded successfully.")

# ========== HELPER FUNCTIONS ==========
def load_image(path):
    img = Image.open(path).convert("L").resize(image_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0)  # shape (1, H, W)

def get_slice_index(fname):
    match = re.search(r"_(\d+)\.png", fname)
    return int(match.group(1)) if match else -1

def get_prefix(fname):
    # Extract everything before the final "_####.png"
    return re.sub(r"_\d+\.png$", "", fname)

# ========== INFERENCE ==========
with torch.no_grad():
    for patient_id in sorted(os.listdir(input_root)):
        patient_dir = os.path.join(input_root, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        out_dir = os.path.join(output_root, patient_id)
        os.makedirs(out_dir, exist_ok=True)

        # Get and sort all patient slices by index
        files = sorted(
            [f for f in os.listdir(patient_dir) if f.endswith(".png")],
            key=get_slice_index
        )

        if len(files) < 2:
            print(f"⚠️ Skipping {patient_id}: not enough slices.")
            continue

        # === Copy all original slices first ===
        for f in files:
            src = os.path.join(patient_dir, f)
            dst = os.path.join(out_dir, f)
            shutil.copy2(src, dst)
        print(f"📁 Copied {len(files)} original slices for {patient_id}")

        # === Generate intermediate slices ===
        for i in range(len(files) - 1):
            f1, f2 = files[i], files[i + 1]
            idx1, idx2 = get_slice_index(f1), get_slice_index(f2)
            prefix = get_prefix(f1)  # e.g. "ID0001_axial"

            # Compute mid index (e.g. (1 + 9)//2 = 5)
            mid_idx = (idx1 + idx2) // 2

            # Prepare input tensor
            s1 = load_image(os.path.join(patient_dir, f1))
            s2 = load_image(os.path.join(patient_dir, f2))
            inp = torch.cat([s1, s2], dim=0).unsqueeze(0).to(device)  # shape (1, 2, H, W)

            # Predict mid-slice
            pred = model(inp).cpu().squeeze(0)  # shape (1, H, W)

            # Save output (consistent naming pattern)
            out_name = f"{prefix}_{mid_idx:04d}.png"
            out_path = os.path.join(out_dir, out_name)
            save_image(pred, out_path)
            print(f"🧩 Generated: {out_path}")

print("✅ All original and intermediate slices saved successfully.")
