import os
import shutil
import pandas as pd

# === Configuration ===
csv_path = "output_filtered_paths.csv"           # Path to your CSV file
column_name = "file_path"       # The column name containing file paths
destination_folder = r"PKG - Yale-Brain-Mets-Longitudinal\PKG - Yale-Brain-Mets-Longitudinal\High_res_images"  # Folder where files will be copied

# === Create destination folder if it doesn't exist ===
os.makedirs(destination_folder, exist_ok=True)

# === Read CSV ===
df = pd.read_csv(csv_path)

# === Loop through each file path ===
for i, src_path in enumerate(df[column_name]):
    if os.path.isfile(src_path):
        # Keep original filename
        filename = os.path.basename(src_path)
        dst_path = os.path.join(destination_folder, filename)
        
        # Copy file
        shutil.copy2(src_path, dst_path)
        print(f"[{i+1}/{len(df)}] Copied: {filename}")
    else:
        print(f"[{i+1}/{len(df)}] File not found: {src_path}")

print("✅ Copying complete!")
