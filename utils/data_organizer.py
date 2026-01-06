import cv2
import numpy as np
import os
import glob
import shutil
import argparse
from tqdm import tqdm

def is_skin_hsv(img):
    """
    Detects if the image is watermelon skin or flesh using HSV color space.
    
    Args:
        img (numpy.ndarray): The input image in BGR format.
        
    Returns:
        bool: True if the image is skin (green dominant), False if flesh (red dominant).
    """
    if img is None:
        return False

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define Green range (Skin)
    # Hue: 30-90 covers yellowish-green to dark green
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])

    # Define Red range (Flesh)
    # Red wraps around 0/180 in HSV, so we need two ranges
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 30, 30])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Count pixels
    green_pixels = np.count_nonzero(mask_green)
    red_pixels = np.count_nonzero(mask_red)

    # Determine dominant color
    return green_pixels > red_pixels

def organize_dataset(input_dir, output_dir):
    """
    Organizes the raw dataset into a structured format with Skin/Flesh subfolders.
    
    Args:
        input_dir (str): Path to the raw dataset containing mixed images.
        output_dir (str): Path where the organized dataset will be saved.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Find all watermelon folders (assuming they start with 'watermelon')
    folders = glob.glob(os.path.join(input_dir, "watermelon*"))
    
    print(f"Found {len(folders)} watermelon folders. Processing...")

    for folder_path in tqdm(folders, desc="Organizing Data"):
        folder_name = os.path.basename(folder_path)
        
        # Create destination folder structure
        dest_path = os.path.join(output_dir, folder_name)
        os.makedirs(dest_path, exist_ok=True)
        
        # Create subfolders
        os.makedirs(os.path.join(dest_path, "Skin"), exist_ok=True)
        os.makedirs(os.path.join(dest_path, "Flesh"), exist_ok=True)

        # 1. Copy non-image files (Audio, Metadata, etc.)
        # Copying .csv, .m4a, .txt to the root of the watermelon folder
        all_files = glob.glob(os.path.join(folder_path, "*.*"))
        for f in all_files:
            if f.lower().endswith(('.csv', '.m4a', '.txt')):
                shutil.copy2(f, dest_path)

        # 2. Process and sort images
        images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.png")) + \
                 glob.glob(os.path.join(folder_path, "*.jpeg"))
        
        for img_path in images:
            filename = os.path.basename(img_path)
            
            # Read image
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            # Classify and Copy
            if is_skin_hsv(img):
                shutil.copy2(img_path, os.path.join(dest_path, "Skin", filename))
            else:
                shutil.copy2(img_path, os.path.join(dest_path, "Flesh", filename))

    print(f"\n✅ Processing Complete!")
    print(f"Organized dataset saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize Watermelon Images into Skin/Flesh folders based on color.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset (source).")
    parser.add_argument("--output", type=str, required=True, help="Path to save organized dataset (destination).")
    
    args = parser.parse_args()
    
    organize_dataset(args.input, args.output)