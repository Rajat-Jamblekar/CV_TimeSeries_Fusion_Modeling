import pandas as pd
import numpy as np
from pathlib import Path

# CONFIG
INPUT_FOLDER = r"C:\Users\Dhivya Shankar\Desktop\BITSSem4\Project\data\pose_data"
# Landmarks to swap (Left ID <-> Right ID)
SWAP_PAIRS = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]

def mirror_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    mirrored_df = df.copy()
    
    # 1. Flip all X coordinates (X is normalized 0-1)
    x_cols = [c for c in df.columns if '_x' in c]
    mirrored_df[x_cols] = 1.0 - df[x_cols]
    
    # 2. Swap Left and Right Joint Data
    # Note: Updated 'vis' to 'visibility' based on your CSV header
    for left_id, right_id in SWAP_PAIRS:
        for axis in ['x', 'y', 'z', 'visibility']:
            left_col = f"lm_{left_id}_{axis}"
            right_col = f"lm_{right_id}_{axis}"
            
            # Defensive check: only swap if both columns actually exist
            if left_col in df.columns and right_col in df.columns:
                # We pull from the already X-flipped mirrored_df to maintain consistency
                temp_left = mirrored_df[left_col].copy()
                mirrored_df[left_col] = mirrored_df[right_col]
                mirrored_df[right_col] = temp_left
            
    mirrored_df.to_csv(output_path, index=False)

# Run for all files
csv_files = list(Path(INPUT_FOLDER).glob("*_biomech.csv"))
for csv_path in csv_files:
    if "_mirrored" in csv_path.stem: 
        continue # Skip if already mirrored
        
    out_path = csv_path.parent / f"{csv_path.stem}_mirrored.csv"
    mirror_csv(csv_path, out_path)
    print(f"Processed: {csv_path.name} -> {out_path.name}")

print(f"\nAugmentation complete. Total files in folder: {len(list(Path(INPUT_FOLDER).glob('*_biomech.csv')))}")