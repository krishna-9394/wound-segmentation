import os
import glob
import shutil
import random

# Define paths
data_dir = "./data/dataset/"  # Base directory containing images and labels
image_dir = os.path.join(data_dir, "images")
label_dir = os.path.join(data_dir, "labels")

train_dir = os.path.join(data_dir, "training")
test_dir = os.path.join(data_dir, "testing")

# Subdirectories for images and labels
train_image_dir = os.path.join(train_dir, "images")
train_label_dir = os.path.join(train_dir, "labels")
test_image_dir = os.path.join(test_dir, "images")
test_label_dir = os.path.join(test_dir, "labels")

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Get all image and label files
image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
label_files = sorted(glob.glob(os.path.join(label_dir, "*.png")))

# Ensure corresponding labels exist
assert len(image_files) == len(label_files), "Mismatch between images and labels"

# Shuffle data randomly (to ensure randomness in the split)
data = list(zip(image_files, label_files))
random.shuffle(data)

# Split data into 80% training, 20% testing
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# Move images and labels to respective directories
for image_path, label_path in train_data:
    shutil.copy(image_path, os.path.join(train_image_dir, os.path.basename(image_path)))
    shutil.copy(label_path, os.path.join(train_label_dir, os.path.basename(label_path)))

for image_path, label_path in test_data:
    shutil.copy(image_path, os.path.join(test_image_dir, os.path.basename(image_path)))
    shutil.copy(label_path, os.path.join(test_label_dir, os.path.basename(label_path)))

print("Dataset successfully split into training and testing sets!")
