import os
import numpy as np
import nibabel as nib
import imageio
from scipy.ndimage import zoom

# === Configuration ===
input_folder = r"PKG - Yale-Brain-Mets-Longitudinal\PKG - Yale-Brain-Mets-Longitudinal\High_res_images"
output_root = r"Slices"
resized_folder = r"Resized_3D"
target_shape = (256, 256, 256)  # Desired shape for all 3D volumes

# === Create output directories ===
axial_dir = os.path.join(output_root, "axial")
coronal_dir = os.path.join(output_root, "coronal")
sagittal_dir = os.path.join(output_root, "sagittal")
os.makedirs(axial_dir, exist_ok=True)
os.makedirs(coronal_dir, exist_ok=True)
os.makedirs(sagittal_dir, exist_ok=True)
os.makedirs(resized_folder, exist_ok=True)

# === Helper functions ===
def resize_volume(volume, target_shape):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, zoom_factors, order=1)  # linear interpolation

def normalize_volume(volume):
    vol_min = np.min(volume)
    vol_max = np.max(volume)
    if vol_max > vol_min:
        volume = (volume - vol_min) / (vol_max - vol_min)
    else:
        volume = np.zeros_like(volume)
    return (volume * 255).astype(np.uint8)

# === Process all 3D images ===
image_files = [f for f in os.listdir(input_folder) if f.endswith((".nii", ".nii.gz"))]
print(f"Found {len(image_files)} 3D images to process.\n")

for uid, filename in enumerate(sorted(image_files), start=1):
    img_path = os.path.join(input_folder, filename)
    print(f"[{uid}/{len(image_files)}] Processing {filename}...")

    # Load image
    img = nib.load(img_path)
    data = img.get_fdata()
    print(f"   ➤ Original dimensions: {data.shape}")

    # Skip cropping/padding — keep full data
    data_resized = resize_volume(data, target_shape)
    print(f"   ➤ Resized dimensions: {data_resized.shape}")

    # Normalize the entire 3D volume to 0-255
    data_resized = normalize_volume(data_resized)

    # Save resized 3D volume
    out_resized_path = os.path.join(resized_folder, f"ID{uid:04d}_resized.nii.gz")
    nib.save(nib.Nifti1Image(data_resized, affine=np.eye(4)), out_resized_path)

    # Save slices (no further normalization), only if slice has non-zero pixels
    for i in range(data_resized.shape[2]):  # axial
        slice_img = data_resized[:, :, i]
        if np.any(slice_img):
            imageio.imwrite(os.path.join(axial_dir, f"ID{uid:04d}_axial_{i:04d}.png"), slice_img)

    for i in range(data_resized.shape[1]):  # coronal
        slice_img = data_resized[:, i, :]
        if np.any(slice_img):
            imageio.imwrite(os.path.join(coronal_dir, f"ID{uid:04d}_coronal_{i:04d}.png"), slice_img)

    for i in range(data_resized.shape[0]):  # sagittal
        slice_img = data_resized[i, :, :]
        if np.any(slice_img):
            imageio.imwrite(os.path.join(sagittal_dir, f"ID{uid:04d}_sagittal_{i:04d}.png"), slice_img)

print("\n✅ All 3D images resized, normalized, and slices saved (no cropping or padding).")
