import pandas as pd
import numpy as np
import os
from pathlib import Path

INPUT_FOLDER = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\pose_data"
SEQ_OUTPUT = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\sequences"
MAX_FRAMES = 90 
os.makedirs(SEQ_OUTPUT, exist_ok=True)

# 14 Landmarks
KEY_LMS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0, 24]

def extract_sequence(path):
    df = pd.read_csv(path)
    seq = []
    for _, row in df.iterrows():
        frame_data = []
        for lm in KEY_LMS:
            # We use .get() or check if column exists to avoid errors
            frame_data.extend([row[f"lm_{lm}_x"], row[f"lm_{lm}_y"]])
        seq.append(frame_data)
    
    seq = np.array(seq)
    if len(seq) > MAX_FRAMES: 
        seq = seq[:MAX_FRAMES]
    else: 
        seq = np.vstack((seq, np.zeros((MAX_FRAMES-len(seq), len(KEY_LMS)*2))))
    return seq

# --- UPDATED LOOP ---
# We look for ALL csv files in the folder
all_csvs = list(Path(INPUT_FOLDER).glob("*.csv"))

# Filter to include both original biomech and mirrored files
target_files = [f for f in all_csvs if f.name.endswith('_biomech.csv') or f.name.endswith('_mirrored.csv')]

print(f"Found {len(target_files)} pose files (Original + Mirrored). Starting conversion...")

for csv_path in target_files:
    # Get the unique name (e.g., 'shot1' or 'shot1_mirrored')
    shot_id = csv_path.stem.replace("_biomech", "")
    
    # Process and save as .npy
    sequence_data = extract_sequence(csv_path)
    save_path = os.path.join(SEQ_OUTPUT, f"{shot_id}.npy")
    np.save(save_path, sequence_data)

print(f"Done! {len(target_files)} sequences generated in {SEQ_OUTPUT}")