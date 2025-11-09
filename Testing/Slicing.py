import os
import numpy as np
import nibabel as nib
import imageio
from scipy.ndimage import zoom

# === Configuration ===
input_folder = r"Path/to/downloaded/images"
output_root = r"Path/to/slices"

# === Create output directories ===
axial_dir = os.path.join(output_root, "axial")
os.makedirs(axial_dir, exist_ok=True)

# === Process all 3D images ===
image_files = [f for f in os.listdir(input_folder) if f.endswith((".nii", ".nii.gz"))]
print(f"Found {len(image_files)} 3D images to process.\n")

for uid, filename in enumerate(sorted(image_files), start=1):
    img_path = os.path.join(input_folder, filename)
    print(f"[{uid}/{len(image_files)}] Processing {filename}...")

    # Load image
    img = nib.load(img_path)
    data = img.get_fdata()

    # Save slices (no further normalization), only if slice has non-zero pixels
    for i in range(data.shape[2]):  # axial
        slice_img = data[:, :, i]
        if np.any(slice_img):
            imageio.imwrite(os.path.join(axial_dir, f"ID{uid:04d}_axial_{i:04d}.png"), slice_img)
print("\n All slices saved ")
