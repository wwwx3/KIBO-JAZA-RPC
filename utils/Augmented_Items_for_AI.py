import cv2
import glob
import os
from imgaug import augmenters as iaa
import numpy as np

# ------------------------------
# 1. Define Image Augmentation Pipeline
# ------------------------------
# We use imgaug to create variations of each image.
# This helps improve training performance by making the model more robust.
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),                        # Random horizontal flip (50% chance)
    iaa.Affine(rotate=(-25, 25)),           # Random rotation between -25° and 25°
    iaa.GaussianBlur(sigma=(0, 1.5)),       # Apply Gaussian blur
    iaa.AdditiveGaussianNoise(scale=(5, 20)), # Add random noise
    iaa.Multiply((0.8, 1.2)),               # Adjust brightness (darker → brighter)
    iaa.Affine(scale=(0.8, 1.2)),           # Zoom in/out (scale image)
], random_order=True)  # Apply in random order each time


# ------------------------------
# 2. Load Original Images
# ------------------------------
# We assume all input images are stored in "items/" folder as .png files.
image_paths = glob.glob("items/*.png")

# Create an output folder to store augmented datasets
output_base = "augmented_dataset"
os.makedirs(output_base, exist_ok=True)


# ------------------------------
# 3. Augment Each Image
# ------------------------------
for img_path in image_paths:
    # Read image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        continue

    # Extract item name (e.g., 'apple.png' → 'apple')
    item_name = os.path.splitext(os.path.basename(img_path))[0]

    # Create a folder for this item inside augmented_dataset/
    item_output_dir = os.path.join(output_base, item_name)
    os.makedirs(item_output_dir, exist_ok=True)

    print(f"Generating augmentations for: {item_name}")

    # --------------------------
    # 4. Generate Augmented Images
    # --------------------------
    # Change the range (1000) to however many augmented images
    # you want to generate per original image.
    for i in range(1000):
        # Apply random augmentations
        aug_img = augmenters(image=img)

        # Save augmented image with numbered filenames
        filename = os.path.join(item_output_dir, f"{item_name}_{i:04d}.png")
        cv2.imwrite(filename, aug_img)
