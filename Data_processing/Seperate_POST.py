import csv

csv_path = r"D:\Project_Yale\Slices_2\filename_uid_mapping.csv"  # your CSV
output_txt = r"D:\Project_Yale\Slices_2\POST.txt"

post_ids = []

with open(csv_path, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if "POST" in row["Original_Filename"]:
            post_ids.append(row["Resized_Filename"][:6])

# Save to txt
with open(output_txt, "w") as f:
    for pid in post_ids:
        f.write(pid + "\n")

print(f"✅ Saved {len(post_ids)} POST IDs to {output_txt}")
