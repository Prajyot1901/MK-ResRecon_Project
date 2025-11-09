# ===================== split_prefixes.py =====================
import os
import random
from sklearn.model_selection import train_test_split

# --- Configuration ---
seed = 42
test_prefixes_count = 80  # same as in training script
input_txt = r"D:\Project_Yale\Slices_2\POST.txt"  # input file with all prefixes
train_out = r"D:\Project_Yale\Slices_2\train_prefixes.txt"
test_out = r"D:\Project_Yale\Slices_2\test_prefixes.txt"

# --- Set random seed for reproducibility ---
random.seed(seed)

# --- Load all prefixes ---
with open(input_txt, "r") as f:
    prefixes = [line.strip() for line in f if line.strip()]

if not prefixes:
    raise ValueError(f"No prefixes found in {input_txt}")

print(f"✅ Loaded {len(prefixes)} prefixes from {input_txt}")

# --- Compute test ratio as in main script ---
test_ratio = test_prefixes_count / max(1, len(prefixes))

# --- Split the data deterministically ---
train_prefixes, test_prefixes = train_test_split(
    prefixes, test_size=test_ratio, random_state=seed
)

print(f"✅ Training prefixes: {len(train_prefixes)}")
print(f"✅ Testing prefixes: {len(test_prefixes)}")

# --- Save results ---
os.makedirs(os.path.dirname(train_out), exist_ok=True)

with open(train_out, "w") as f:
    for prefix in train_prefixes:
        f.write(prefix + "\n")

with open(test_out, "w") as f:
    for prefix in test_prefixes:
        f.write(prefix + "\n")

print(f"💾 Saved training prefixes to: {train_out}")
print(f"💾 Saved testing prefixes to: {test_out}")
print("🎯 Split completed successfully and matches the training script logic.")
