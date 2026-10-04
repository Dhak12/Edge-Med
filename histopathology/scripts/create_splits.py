from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path("data")
SPLIT_DIR = Path("splits")

SPLIT_DIR.mkdir(exist_ok=True)

# Class mapping
CLASS_NAMES = {
    "normal": 0,
    "benign": 1,
    "malignant": 2
}

# Collect all images
records = []

for class_name, label in CLASS_NAMES.items():
    class_dir = DATA_DIR / class_name

    for image_path in sorted(class_dir.glob("*.png")):
        records.append({
            "path": str(image_path),
            "label": label,
            "class_name": class_name
        })

df = pd.DataFrame(records)

print("Total images:", len(df))
print("\nOriginal class distribution:")
print(df["class_name"].value_counts())

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Second split: temporary → 15% validation, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Shuffle each split
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

# Print results
print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain distribution:")
print(train_df["class_name"].value_counts())

print("\nValidation distribution:")
print(val_df["class_name"].value_counts())

print("\nTest distribution:")
print(test_df["class_name"].value_counts())

print("\nSaved:")
print(SPLIT_DIR / "train.csv")
print(SPLIT_DIR / "val.csv")
print(SPLIT_DIR / "test.csv")