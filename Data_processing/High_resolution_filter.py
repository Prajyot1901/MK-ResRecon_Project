import pandas as pd
import os

# Define your root directory (change this to your actual root path)
root_dir = r"D:\Project_Yale\PKG - Yale-Brain-Mets-Longitudinal\PKG - Yale-Brain-Mets-Longitudinal\Yale-Brain-Mets-Longitudinal"

# Load your CSV file
df = pd.read_csv("your_file.csv")

# Apply condition: only keep rows with slice_thickness < 1.5
filtered = df[df["slice_thickness (mm)"] < 1.5].copy()

# Construct the full path
filtered["file_path"] = filtered.apply(
    lambda row: os.path.join(
        root_dir,
        row["patient_id"],
        row["study_datetime"].split("_")[0],  # extract just YYYY-MM-DD
        row["file_name"]
    ),
    axis=1
)

# Save filtered results to a new CSV
filtered.to_csv("output_filtered_paths.csv", index=False)

print("Filtered file with paths saved as output_filtered_paths.csv")
