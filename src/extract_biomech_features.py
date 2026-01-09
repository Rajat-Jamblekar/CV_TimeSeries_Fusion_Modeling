import pandas as pd
import numpy as np
import os

# ---------------------------
# CONFIG
# ---------------------------
INPUT_CSV = "shot_02_biomech.csv"
OUTPUT_CSV = "shot_02_features.csv"
SHOT_ID = "shot_02"

# Landmark indices (MediaPipe)
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# ---------------------------
# Helper Functions
# ---------------------------
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_angle(a, b, c):
    """
    Calculate angle at point b (in degrees) for points a-b-c
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv(INPUT_CSV)

# ---------------------------
# Feature Containers
# ---------------------------
elbow_angles = []
wrist_heights = []
stance_widths = []

# ---------------------------
# Frame-wise Feature Computation
# ---------------------------
for _, row in df.iterrows():

    shoulder = np.array([
        row[f"lm_{RIGHT_SHOULDER}_x"],
        row[f"lm_{RIGHT_SHOULDER}_y"]
    ])

    elbow = np.array([
        row[f"lm_{RIGHT_ELBOW}_x"],
        row[f"lm_{RIGHT_ELBOW}_y"]
    ])

    wrist = np.array([
        row[f"lm_{RIGHT_WRIST}_x"],
        row[f"lm_{RIGHT_WRIST}_y"]
    ])

    left_ankle = np.array([
        row[f"lm_{LEFT_ANKLE}_x"],
        row[f"lm_{LEFT_ANKLE}_y"]
    ])

    right_ankle = np.array([
        row[f"lm_{RIGHT_ANKLE}_x"],
        row[f"lm_{RIGHT_ANKLE}_y"]
    ])

    # Elbow angle
    angle = calculate_angle(shoulder, elbow, wrist)
    elbow_angles.append(angle)

    # Wrist height (Y-axis, inverted in image coordinates)
    wrist_heights.append(wrist[1])

    # Stance width
    stance_widths.append(euclidean_distance(left_ankle, right_ankle))

# ---------------------------
# Aggregate Features
# ---------------------------
features = {
    "shot_id": SHOT_ID,
    "mean_elbow_angle": np.mean(elbow_angles),
    "max_wrist_height": np.max(wrist_heights),
    "mean_stance_width": np.mean(stance_widths),
    "shot_duration_frames": len(df)
}

# ---------------------------
# Save Output
# ---------------------------
features_df = pd.DataFrame([features])
features_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Biomechanical feature extraction complete")
print(features_df)
