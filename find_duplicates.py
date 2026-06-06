#!/usr/bin/env python3
"""
YOLO OBB Duplicate Image Scanner
Authored by Antigravity

This script indexes all images in the train and val splits under mvp_1/images
and calculates a 64-bit horizontal Perceptual Difference Hash (dHash) for each.
It then cross-compares all hashes to detect:
1. Cross-set duplicates (data leakage between train and val splits).
2. Intra-set duplicates (redundant images within the same split).

Hamming Distance Thresholds:
- 0: Exact duplicates.
- 1 to 4: Near-duplicates (extremely similar, likely scaled, re-compressed, or watermarked copies).
"""

import os
import sys
import time
from PIL import Image
import numpy as np

# Configurations
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MVP_DIR = os.path.join(WORKSPACE_DIR, "mvp_1")
IMAGES_DIR = os.path.join(MVP_DIR, "images")

# Hamming Distance Threshold (0-64 bits). 4 or lower is highly reliable for duplicates.
HAMMING_THRESHOLD = 4

def dhash(image_path, hash_size=8):
    """
    Computes a 64-bit horizontal Difference Hash (dHash) for a given image.
    This is extremely robust to resizing, aspect ratio shifts, compression, and watermarks.
    """
    try:
        with Image.open(image_path) as img:
            # Downsample to (9, 8) in grayscale
            # We use bilinear resampling for a smooth, fast downscale
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img)
            
        # Compare adjacent horizontal pixels (size: 8x8)
        diff = pixels[:, :-1] > pixels[:, 1:]
        
        # Pack 64 boolean diff values into a 16-character hex string
        decimal_val = 0
        hex_string = []
        for i, val in enumerate(diff.flatten()):
            if val:
                decimal_val += 2 ** (i % 8)
            if (i % 8) == 7:
                hex_string.append(hex(decimal_val)[2:].zfill(2))
                decimal_val = 0
        return "".join(hex_string)
    except Exception as e:
        print(f"[!] Error processing {image_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Calculates the number of differing bits between two 64-bit hex hashes."""
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    return bin(int1 ^ int2).count("1")

def scan_dataset():
    print("==================================================")
    print("YOLO OBB Perceptual Duplicate Scanner")
    print("==================================================")
    
    start_time = time.time()
    
    if not os.path.exists(IMAGES_DIR):
        print(f"[!] Error: Images directory not found at {IMAGES_DIR}")
        print("    Please run 'prepare_dataset.py' first.")
        sys.exit(1)
        
    # Index images
    image_records = []  # List of dicts containing split, filename, path, hash
    splits = ["train", "val"]
    
    for split in splits:
        split_dir = os.path.join(IMAGES_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        files = [f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[*] Indexing split '{split}': Found {len(files)} images...")
        
        for file in files:
            path = os.path.join(split_dir, file)
            img_hash = dhash(path)
            if img_hash:
                image_records.append({
                    "split": split,
                    "filename": file,
                    "path": path,
                    "hash": img_hash
                })
                
    total_indexed = len(image_records)
    print(f"\n[+] Successfully computed hashes for {total_indexed} images (Time taken: {time.time() - start_time:.2f}s)")
    
    # Perform comparisons
    print("\n[*] Starting similarity scans (threshold <= {} bits difference)...".format(HAMMING_THRESHOLD))
    
    cross_duplicates = []
    intra_train_duplicates = []
    intra_val_duplicates = []
    
    # To prevent listing a match (A, B) and then its symmetrical counterpart (B, A)
    seen_pairs = set()
    
    for i in range(total_indexed):
        rec1 = image_records[i]
        for j in range(i + 1, total_indexed):
            rec2 = image_records[j]
            
            # Symmetrical key
            pair_key = tuple(sorted([rec1["path"], rec2["path"]]))
            if pair_key in seen_pairs:
                continue
                
            dist = hamming_distance(rec1["hash"], rec2["hash"])
            
            if dist <= HAMMING_THRESHOLD:
                seen_pairs.add(pair_key)
                match_record = {
                    "img1_split": rec1["split"],
                    "img1_file": rec1["filename"],
                    "img1_path": rec1["path"],
                    "img2_split": rec2["split"],
                    "img2_file": rec2["filename"],
                    "img2_path": rec2["path"],
                    "distance": dist
                }
                
                # Categorize matches
                if rec1["split"] != rec2["split"]:
                    cross_duplicates.append(match_record)
                elif rec1["split"] == "train":
                    intra_train_duplicates.append(match_record)
                else:
                    intra_val_duplicates.append(match_record)
                    
    # Generate report
    print("\n==================================================")
    print("Duplicate Scanner Report")
    print("==================================================")
    
    # 1. Cross-set data leakage
    print(f"[*] 1. CROSS-SPLIT DUPLICATES (Data Leakage: Train vs Val): {len(cross_duplicates)}")
    if cross_duplicates:
        for idx, match in enumerate(cross_duplicates):
            print(f"    [{idx+1}] Train: {match['img1_file'] if match['img1_split'] == 'train' else match['img2_file']}")
            print(f"        Val  : {match['img2_file'] if match['img1_split'] == 'train' else match['img1_file']}")
            print(f"        Hamming Distance: {match['distance']} (bits difference)")
    else:
        print("    [+] Excellent! No data leakage found between train and validation sets.")
        
    # 2. Intra-set redundancy
    print(f"\n[*] 2. INTRA-SET REDUNDANCY (Duplicates within splits):")
    print(f"    - Train split redundancy: {len(intra_train_duplicates)}")
    if intra_train_duplicates:
        for idx, match in enumerate(intra_train_duplicates[:10]):  # Cap display to 10
            print(f"        [{idx+1}] Match: {match['img1_file']} <==> {match['img2_file']} (Dist: {match['distance']})")
        if len(intra_train_duplicates) > 10:
            print(f"        ... and {len(intra_train_duplicates) - 10} more training matches.")
            
    print(f"    - Val split redundancy  : {len(intra_val_duplicates)}")
    if intra_val_duplicates:
        for idx, match in enumerate(intra_val_duplicates):
            print(f"        [{idx+1}] Match: {match['img1_file']} <==> {match['img2_file']} (Dist: {match['distance']})")
            
    print("\n==================================================")
    
    # Recommendations
    if cross_duplicates:
        print("[!] ACTION RECOMMENDED:")
        print("    We found duplicates spanning your training and validation sets!")
        print("    To prevent optimistic validation bias (data leakage), you should remove")
        print("    these specific duplicate images from your validation set (val/split) and")
        print("    their corresponding label txt files.")
    elif intra_train_duplicates or intra_val_duplicates:
        print("[*] RECOMMENDATION:")
        print("    No cross-split leakage found, but we found redundant images in individual splits.")
        print("    You can safely leave them or prune them to reduce training overhead.")
    else:
        print("[+] SUCCESS: Your dataset splits are extremely clean and free from duplicate plant images!")

if __name__ == "__main__":
    scan_dataset()
