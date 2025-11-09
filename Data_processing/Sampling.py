import os
import shutil
import re
from concurrent.futures import ThreadPoolExecutor

# Set your input and output directories
input_dir = r"D:\Project_Yale\Slices\axial"
output_dir = r"D:\Project_Yale\Slices_Original_8\axial"

# Regular expression to parse filenames like: ID0001_axial_0020.png
pattern = re.compile(r"^(?P<patient_id>[A-Za-z0-9]{6})_axial_(?P<index>\d{4})\.png$")

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_file(filename):
    match = pattern.match(filename)
    if not match:
        return  # Skip files that don't match the pattern
    
    patient_id = match.group("patient_id")
    slice_index = int(match.group("index"))
    
    # Check if slice index matches 8*i + 1
    if (slice_index - 1) % 8 == 0:
        src_path = os.path.join(input_dir, filename)
        patient_folder = os.path.join(output_dir, patient_id)
        os.makedirs(patient_folder, exist_ok=True)
        dst_path = os.path.join(patient_folder, filename)
        shutil.copy2(src_path, dst_path)

def main():
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    # Use ThreadPoolExecutor to parallelize copying
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_file, all_files)
    print("Filtering complete.")

if __name__ == "__main__":
    main()
