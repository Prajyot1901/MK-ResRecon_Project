import os
import re
import warnings
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

# ---------------------- CONFIGURATION ---------------------- #
root_dir = r"Path/to/predicted/slices"     # <-- change this
out_dir  = r"Path/to/output/directory"   # <-- change this
use_gpu  = True                          # set False to force CPU
normalize = False                        # normalize each 2D slice to [0,1]
target_shape = (256, 256, 256)           # (H, W, D)
dtype = torch.float32
# ------------------------------------------------------------ #

# Regex pattern: extract patient ID (first 6 chars) and slice index
FNAME_RE = re.compile(r'(?P<pid>.{6})_axial_(?P<idx>\d+)', re.IGNORECASE)

def process_patient_folder(patient_folder: Path,
                           out_dir: Path,
                           device: torch.device,
                           target_shape=(256,256,256),
                           dtype=torch.float32,
                           normalize=False):

    H, W, D = target_shape
    vol = torch.zeros((H, W, D), dtype=dtype, device=device)

    imgs = [p for p in patient_folder.iterdir() if p.is_file()]
    if not imgs:
        warnings.warn(f"No files in {patient_folder}; skipping.")
        return

    index_map = {}
    patient_id = None
    for p in imgs:
        m = FNAME_RE.search(p.stem)
        if not m:
            continue
        pid = m.group('pid')
        idx = int(m.group('idx'))
        if patient_id is None:
            patient_id = pid
        if 0 <= idx < D:
            index_map[idx] = p

    if patient_id is None:
        warnings.warn(f"No valid image files in {patient_folder}")
        return

    for idx in tqdm(sorted(index_map.keys()), desc=f"{patient_id}", leave=False):
        p = index_map[idx]
        img = Image.open(p).convert("L")
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)

        if normalize:
            mn, mx = arr.min(), arr.max()
            arr = (arr - mn) / (mx - mn) if mx > mn else arr * 0.0

        t = torch.from_numpy(arr).to(device=device, dtype=dtype)
        vol[:, :, idx] = t

    vol_cpu = vol.detach().cpu().numpy()
    affine = np.eye(4, dtype=np.float32)
    nifti_img = nib.Nifti1Image(vol_cpu, affine)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient_id}.nii.gz"
    nib.save(nifti_img, str(out_path))
    print(f"Saved {out_path}")

def main():
    root = Path(root_dir)
    out = Path(out_dir)
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")

    entries = [p for p in root.iterdir() if p.is_dir()]
    files_in_root = [p for p in root.iterdir() if p.is_file()]
    if files_in_root:
        entries.insert(0, root)

    if not entries:
        raise RuntimeError(f"No folders or files found in {root}")

    print(f"Found {len(entries)} patient folder(s).")

    for folder in tqdm(entries, desc="Patients"):
        try:
            process_patient_folder(folder, out, device=device, normalize=normalize)
        except Exception as e:
            print(f"FAILED for {folder}: {e}")

if __name__ == "__main__":
    main()
