import os
import glob
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# ==========================================
# Calibration Settings (High Contrast)
# ==========================================
RAW_MIN = 150
RAW_MAX = 183

TARGET_MIN = 90
TARGET_MAX = 245

def map_value(value, in_min, in_max, out_min, out_max):
    """
    Maps a value from one range to another with clamping.
    """
    if value < in_min: value = in_min
    if value > in_max: value = in_max
    mapped = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return int(mapped)

def get_quality_grade(score):
    """
    Converts the redness score to a quality grade (1-5).
    1: Least red (Unripe/White) | 5: Most red (Excellent)
    """
    if score >= 220: return 5  # Excellent
    if score >= 185: return 4  # Good
    if score >= 150: return 3  # Medium
    if score >= 115: return 2  # Weak
    return 1                   # Very Weak

def calculate_lab_redness(img):
    """
    Calculates redness using the 'a' channel of CIELab color space.
    Uses HSV masking to ignore background/rind.
    """
    # Convert to HSV for masking red areas
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define red ranges in HSV
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    # If no red detected, return 0
    if np.count_nonzero(mask) == 0: return 0

    # Convert to Lab and extract 'a' channel (Green-Red component)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_channel = lab_img[:, :, 1]
    
    # Calculate mean of 'a' channel only in the masked area
    mean_a = cv2.mean(a_channel, mask=mask)[0]
    return mean_a

def process_dataset(dataset_path):
    """
    Iterates through the dataset and renames Flesh folders based on redness score.
    Target format: Flesh_{Score}_{Grade}
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Path '{dataset_path}' not found.")
        return

    # Find all watermelon folders
    root_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and "watermelon" in f]
    
    print(f"Starting Redness Grading for {len(root_folders)} watermelons...")
    
    count_processed = 0

    for root_folder in tqdm(root_folders, desc="Grading"):
        root_path = os.path.join(dataset_path, root_folder)

        # Find the Flesh folder (handling existing names like Flesh_120_2)
        flesh_candidates = glob.glob(os.path.join(root_path, "Flesh*"))
        if not flesh_candidates: continue

        current_flesh_path = flesh_candidates[0]
        
        # Find images in Flesh folder
        img_files = glob.glob(os.path.join(current_flesh_path, "*.jpg")) + \
                    glob.glob(os.path.join(current_flesh_path, "*.png"))
        
        if not img_files: continue

        # Calculate raw scores for all images in the folder
        raw_scores = []
        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is None: continue
            
            s = calculate_lab_redness(img)
            if s > 0: raw_scores.append(s)

        if not raw_scores: continue

        avg_raw_lab = np.mean(raw_scores)

        # Apply calibration
        final_score = map_value(avg_raw_lab, RAW_MIN, RAW_MAX, TARGET_MIN, TARGET_MAX)

        # Determine grade (1-5)
        grade = get_quality_grade(final_score)

        # Rename folder
        new_folder_name = f"Flesh_{final_score}_{grade}"
        new_full_path = os.path.join(root_path, new_folder_name)

        if current_flesh_path != new_full_path:
            try:
                os.rename(current_flesh_path, new_full_path)
                count_processed += 1
            except Exception as e:
                print(f"Error renaming {current_flesh_path}: {e}")

    print(f"\nProcessing Complete. {count_processed} folders renamed/updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate redness score and rename Flesh folders.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the processed dataset.")
    
    args = parser.parse_args()
    
    process_dataset(args.dataset)