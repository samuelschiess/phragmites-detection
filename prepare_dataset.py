#!/usr/bin/env python3
"""
YOLO OBB Dataset Restructuring and Split Utility
Authored by Antigravity

This script reorganizes the mvp_1 folder to align with the standard YOLO directory structure:
- Splits the 387 annotated images/labels into Train (85%) and Val (15%).
- Splits the 24 RandomBackground images into Train and Val, and automatically generates
  corresponding 0-byte .txt label files so YOLO processes them correctly as backgrounds.
- Generates a fresh, corrected data.yaml pointing to these split folders via absolute paths.
- Cleans up deprecated files (like train.txt and the old RandomBackround folder).
"""

import os
import shutil
import random
import yaml  # In case we want to write data.yaml using yaml, or we can write it as plain text

# Set random seed for reproducibility
random.seed(42)

# Configurations
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MVP_DIR = os.path.join(WORKSPACE_DIR, "datasets", "mvp_2")

# Inputs
RAW_IMAGES_DIR = os.path.join(MVP_DIR, "images", "train")
RAW_LABELS_DIR = os.path.join(MVP_DIR, "labels", "train")
BG_DIR = os.path.join(MVP_DIR, "RandomBackround")

# Target splits
SPLITS = ["train", "val"]
VAL_RATIO = 0.15  # 15% validation split

def setup_directories():
    """Create val directories if they do not exist."""
    os.makedirs(os.path.join(MVP_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(MVP_DIR, "labels", "val"), exist_ok=True)
    print("[*] Set up validation directories.")

def process_labeled_data():
    """Restructure labeled images and labels by moving 15% of them to val."""
    print("\n[*] Processing labeled oriented bounding box data...")
    
    # List all training images (which are currently all in images/train)
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"[!] Error: Raw images directory not found at {RAW_IMAGES_DIR}")
        return
        
    all_images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_labeled = len(all_images)
    print(f"    - Found {total_labeled} annotated training images.")
    
    # Determine split counts
    val_count = int(total_labeled * VAL_RATIO)
    train_count = total_labeled - val_count
    print(f"    - Destination counts: Train = {train_count}, Val = {val_count} (15% split)")
    
    # Shuffle and select validation set
    random.shuffle(all_images)
    val_images = all_images[:val_count]
    
    moved_count = 0
    for img_name in val_images:
        base_name, _ = os.path.splitext(img_name)
        
        # Paths
        src_img = os.path.join(RAW_IMAGES_DIR, img_name)
        dst_img = os.path.join(MVP_DIR, "images", "val", img_name)
        
        src_lbl = os.path.join(RAW_LABELS_DIR, f"{base_name}.txt")
        dst_lbl = os.path.join(MVP_DIR, "labels", "val", f"{base_name}.txt")
        
        # Verify that both image and label exist
        if os.path.exists(src_img) and os.path.exists(src_lbl):
            # Move image
            shutil.move(src_img, dst_img)
            # Move label
            shutil.move(src_lbl, dst_lbl)
            moved_count += 1
        else:
            print(f"    [!] Warning: Missing pair for {img_name} (Label exists: {os.path.exists(src_lbl)})")
            
    print(f"    - Successfully split and moved {moved_count} labeled image-label pairs to val/.")

def process_background_data():
    """Distribute RandomBackground images into train and val, and create empty labels."""
    print("\n[*] Processing background images...")
    
    if not os.path.exists(BG_DIR):
        print("    [!] No 'RandomBackround' folder found. Backgrounds may already be processed or missing.")
        return
        
    bg_images = [f for f in os.listdir(BG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_bg = len(bg_images)
    print(f"    - Found {total_bg} background images in {BG_DIR}.")
    
    # Distribute backgrounds
    random.shuffle(bg_images)
    
    train_bg_count = 0
    val_bg_count = 0
    
    for bg_name in bg_images:
        base_name, _ = os.path.splitext(bg_name)
        
        # 15% chance to go to val split
        is_val = random.random() < VAL_RATIO
        split = "val" if is_val else "train"
        
        # Paths
        src_img = os.path.join(BG_DIR, bg_name)
        dst_img = os.path.join(MVP_DIR, "images", split, bg_name)
        dst_lbl = os.path.join(MVP_DIR, "labels", split, f"{base_name}.txt")
        
        # Move image
        shutil.move(src_img, dst_img)
        
        # Create empty 0-byte label file
        with open(dst_lbl, "w") as f:
            pass  # Creates empty file
            
        if is_val:
            val_bg_count += 1
        else:
            train_bg_count += 1
            
    print(f"    - Backgrounds distributed: Train = {train_bg_count}, Val = {val_bg_count}")
    
    # Remove empty background directory
    try:
        os.rmdir(BG_DIR)
        print("    - Cleaned up raw background directory.")
    except Exception as e:
        print(f"    [!] Warning cleaning background folder: {e}")

def generate_data_yaml():
    """Write a fresh data.yaml pointing to the correct split folders via absolute paths."""
    print("\n[*] Generating updated data.yaml...")
    yaml_path = os.path.join(MVP_DIR, "data.yaml")
    
    # Convert path to use forward slashes (recommended for YOLO cross-platform compatibility)
    abs_mvp_path = MVP_DIR.replace("\\", "/")
    
    yaml_content = f"""# Phragmites Detection Round 1 MVP Dataset
# Generated by Antigravity prepare_dataset.py

path: {abs_mvp_path}
train: images/train
val: images/val

names:
  0: Invasive Phragmite
  1: Native Phragmites
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"    - Wrote data.yaml to {yaml_path}")

def cleanup_deprecated():
    """Remove obsolete train.txt to keep the repository tidy."""
    print("\n[*] Cleaning up deprecated files...")
    train_txt = os.path.join(MVP_DIR, "train.txt")
    if os.path.exists(train_txt):
        os.remove(train_txt)
        print("    - Removed deprecated train.txt list file.")
    else:
        print("    - No train.txt list file found.")

def verify_dataset():
    """Print out summary statistics and verify integrity."""
    print("\n==================================================")
    print("Dataset Reorganization Verification Summary")
    print("==================================================")
    
    for split in SPLITS:
        img_dir = os.path.join(MVP_DIR, "images", split)
        lbl_dir = os.path.join(MVP_DIR, "labels", split)
        
        images = os.listdir(img_dir) if os.path.exists(img_dir) else []
        labels = os.listdir(lbl_dir) if os.path.exists(lbl_dir) else []
        
        print(f"[*] {split.upper()} SPLIT:")
        print(f"    - Images: {len(images)}")
        print(f"    - Labels: {len(labels)}")
        
        # Verify alignment
        img_bases = {os.path.splitext(f)[0] for f in images}
        lbl_bases = {os.path.splitext(f)[0] for f in labels}
        
        unmatched_imgs = img_bases - lbl_bases
        unmatched_lbls = lbl_bases - img_bases
        
        if unmatched_imgs:
            print(f"    [!] Error: Images without labels: {unmatched_imgs}")
        if unmatched_lbls:
            print(f"    [!] Error: Labels without images: {unmatched_lbls}")
            
        if not unmatched_imgs and not unmatched_lbls:
            print("    [+] Integrity check passed: 1:1 image-to-label alignment verified.")

if __name__ == "__main__":
    setup_directories()
    process_labeled_data()
    process_background_data()
    generate_data_yaml()
    cleanup_deprecated()
    verify_dataset()
