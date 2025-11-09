import os
import csv

# === Paths (same as before) ===
input_folder = r"PKG - Yale-Brain-Mets-Longitudinal\PKG - Yale-Brain-Mets-Longitudinal\High_res_images"
output_root = r"Slices_2"

# === Gather filenames (same sorting logic) ===
image_files = [f for f in os.listdir(input_folder) if f.endswith((".nii", ".nii.gz"))]
image_files_sorted = sorted(image_files)

# === Save CSV mapping ===
csv_path = os.path.join(output_root, "filename_uid_mapping.csv")
os.makedirs(output_root, exist_ok=True)

with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["UID", "Original_Filename", "Resized_Filename"])
    for uid, filename in enumerate(image_files_sorted, start=1):
        resized_filename = f"ID{uid:04d}_resized.nii.gz"
        writer.writerow([uid, filename, resized_filename])

print(f"✅ Mapping saved to: {csv_path}")
