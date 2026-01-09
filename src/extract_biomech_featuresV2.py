import pandas as pd
import numpy as np
import os
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
INPUT_FOLDER = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\pose_data"
OUTPUT_FOLDER = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\features"
COMBINED_OUTPUT = "all_shots_features.csv"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Landmark indices (MediaPipe Pose - 33 landmarks)
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# ---------------------------
# Helper Functions
# ---------------------------
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def euclidean_distance_3d(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_angle(a, b, c):
    """Calculate angle at point b (in degrees) for points a-b-c"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_velocity(values, fps=30):
    """Calculate velocity (change per second)"""
    if len(values) < 2:
        return 0
    velocities = np.diff(values) * fps
    return np.mean(np.abs(velocities))

def calculate_acceleration(values, fps=30):
    """Calculate acceleration (change in velocity)"""
    if len(values) < 3:
        return 0
    velocities = np.diff(values) * fps
    accelerations = np.diff(velocities) * fps
    return np.mean(np.abs(accelerations))

def calculate_hip_rotation(left_hip, right_hip):
    """Calculate hip rotation angle relative to horizontal"""
    vector = right_hip - left_hip
    angle = np.arctan2(vector[1], vector[0])
    return np.degrees(angle)

def calculate_shoulder_rotation(left_shoulder, right_shoulder):
    """Calculate shoulder rotation angle relative to horizontal"""
    vector = right_shoulder - left_shoulder
    angle = np.arctan2(vector[1], vector[0])
    return np.degrees(angle)

def calculate_center_of_mass(landmarks_dict):
    """Estimate center of mass (simplified as midpoint of hips and shoulders)"""
    left_hip = landmarks_dict['left_hip']
    right_hip = landmarks_dict['right_hip']
    left_shoulder = landmarks_dict['left_shoulder']
    right_shoulder = landmarks_dict['right_shoulder']
    
    com = (left_hip + right_hip + left_shoulder + right_shoulder) / 4
    return com

def calculate_body_sway(com_positions):
    """Calculate body sway as standard deviation of COM position"""
    if len(com_positions) < 2:
        return 0, 0
    com_array = np.array(com_positions)
    sway_x = np.std(com_array[:, 0])
    sway_y = np.std(com_array[:, 1])
    return sway_x, sway_y

# ---------------------------
# Feature Extraction Function
# ---------------------------
def extract_features_from_csv(csv_path, shot_id):
    """Extract comprehensive biomechanical features from a single CSV file"""
    
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print(f"⚠️  Warning: {shot_id} has no data")
            return None
        
        # Feature containers
        right_elbow_angles = []
        left_elbow_angles = []
        right_knee_angles = []
        left_knee_angles = []
        
        right_wrist_heights = []
        left_wrist_heights = []
        right_wrist_positions_3d = []
        left_wrist_positions_3d = []
        
        head_positions = []
        head_tilts = []
        
        hip_rotations = []
        shoulder_rotations = []
        torso_angles = []
        
        stance_widths = []
        com_positions = []
        base_of_support = []
        
        left_ankle_positions = []
        right_ankle_positions = []
        
        # Frame-wise computation
        for _, row in df.iterrows():
            # Extract all key landmarks
            nose = np.array([row[f"lm_{NOSE}_x"], row[f"lm_{NOSE}_y"], row[f"lm_{NOSE}_z"]])
            left_eye = np.array([row[f"lm_{LEFT_EYE}_x"], row[f"lm_{LEFT_EYE}_y"]])
            right_eye = np.array([row[f"lm_{RIGHT_EYE}_x"], row[f"lm_{RIGHT_EYE}_y"]])
            
            # Shoulders
            left_shoulder = np.array([row[f"lm_{LEFT_SHOULDER}_x"], row[f"lm_{LEFT_SHOULDER}_y"], row[f"lm_{LEFT_SHOULDER}_z"]])
            right_shoulder = np.array([row[f"lm_{RIGHT_SHOULDER}_x"], row[f"lm_{RIGHT_SHOULDER}_y"], row[f"lm_{RIGHT_SHOULDER}_z"]])
            
            # Arms
            left_elbow = np.array([row[f"lm_{LEFT_ELBOW}_x"], row[f"lm_{LEFT_ELBOW}_y"], row[f"lm_{LEFT_ELBOW}_z"]])
            right_elbow = np.array([row[f"lm_{RIGHT_ELBOW}_x"], row[f"lm_{RIGHT_ELBOW}_y"], row[f"lm_{RIGHT_ELBOW}_z"]])
            left_wrist = np.array([row[f"lm_{LEFT_WRIST}_x"], row[f"lm_{LEFT_WRIST}_y"], row[f"lm_{LEFT_WRIST}_z"]])
            right_wrist = np.array([row[f"lm_{RIGHT_WRIST}_x"], row[f"lm_{RIGHT_WRIST}_y"], row[f"lm_{RIGHT_WRIST}_z"]])
            
            # Hips
            left_hip = np.array([row[f"lm_{LEFT_HIP}_x"], row[f"lm_{LEFT_HIP}_y"], row[f"lm_{LEFT_HIP}_z"]])
            right_hip = np.array([row[f"lm_{RIGHT_HIP}_x"], row[f"lm_{RIGHT_HIP}_y"], row[f"lm_{RIGHT_HIP}_z"]])
            
            # Legs
            left_knee = np.array([row[f"lm_{LEFT_KNEE}_x"], row[f"lm_{LEFT_KNEE}_y"], row[f"lm_{LEFT_KNEE}_z"]])
            right_knee = np.array([row[f"lm_{RIGHT_KNEE}_x"], row[f"lm_{RIGHT_KNEE}_y"], row[f"lm_{RIGHT_KNEE}_z"]])
            left_ankle = np.array([row[f"lm_{LEFT_ANKLE}_x"], row[f"lm_{LEFT_ANKLE}_y"], row[f"lm_{LEFT_ANKLE}_z"]])
            right_ankle = np.array([row[f"lm_{RIGHT_ANKLE}_x"], row[f"lm_{RIGHT_ANKLE}_y"], row[f"lm_{RIGHT_ANKLE}_z"]])
            
            # === ARM FEATURES ===
            right_elbow_angles.append(calculate_angle(right_shoulder, right_elbow, right_wrist))
            left_elbow_angles.append(calculate_angle(left_shoulder, left_elbow, left_wrist))
            
            right_wrist_heights.append(right_wrist[1])
            left_wrist_heights.append(left_wrist[1])
            right_wrist_positions_3d.append(right_wrist)
            left_wrist_positions_3d.append(left_wrist)
            
            # === LEG FEATURES ===
            right_knee_angles.append(calculate_angle(right_hip, right_knee, right_ankle))
            left_knee_angles.append(calculate_angle(left_hip, left_knee, left_ankle))
            
            # === HEAD FEATURES ===
            head_center = (left_eye + right_eye) / 2
            head_positions.append(head_center)
            
            # Head tilt (angle between eyes)
            eye_vector = right_eye - left_eye
            head_tilt = np.arctan2(eye_vector[1], eye_vector[0])
            head_tilts.append(np.degrees(head_tilt))
            
            # === HIP & TORSO FEATURES ===
            hip_rotation = calculate_hip_rotation(left_hip[:2], right_hip[:2])
            hip_rotations.append(hip_rotation)
            
            shoulder_rotation = calculate_shoulder_rotation(left_shoulder[:2], right_shoulder[:2])
            shoulder_rotations.append(shoulder_rotation)
            
            # Torso angle (shoulder midpoint to hip midpoint relative to vertical)
            shoulder_mid = (left_shoulder[:2] + right_shoulder[:2]) / 2
            hip_mid = (left_hip[:2] + right_hip[:2]) / 2
            torso_vector = shoulder_mid - hip_mid
            torso_angle = np.arctan2(torso_vector[0], -torso_vector[1])  # Relative to vertical
            torso_angles.append(np.degrees(torso_angle))
            
            # === STANCE & STABILITY ===
            stance_width = euclidean_distance_3d(left_ankle, right_ankle)
            stance_widths.append(stance_width)
            
            # Center of mass estimation
            landmarks = {
                'left_hip': left_hip,
                'right_hip': right_hip,
                'left_shoulder': left_shoulder,
                'right_shoulder': right_shoulder
            }
            com = calculate_center_of_mass(landmarks)
            com_positions.append(com[:2])  # X, Y only for sway
            
            # Base of support (distance between ankles)
            base_of_support.append(stance_width)
            
            left_ankle_positions.append(left_ankle)
            right_ankle_positions.append(right_ankle)
        
        # === CALCULATE AGGREGATE FEATURES ===
        
        # Body sway
        sway_x, sway_y = calculate_body_sway(com_positions)
        
        # Ankle stability (movement range)
        left_ankle_array = np.array(left_ankle_positions)
        right_ankle_array = np.array(right_ankle_positions)
        left_ankle_movement = np.std(left_ankle_array, axis=0)
        right_ankle_movement = np.std(right_ankle_array, axis=0)
        
        # Wrist path length (total distance traveled)
        right_wrist_path = sum([euclidean_distance_3d(right_wrist_positions_3d[i], right_wrist_positions_3d[i+1]) 
                                for i in range(len(right_wrist_positions_3d)-1)])
        left_wrist_path = sum([euclidean_distance_3d(left_wrist_positions_3d[i], left_wrist_positions_3d[i+1]) 
                               for i in range(len(left_wrist_positions_3d)-1)])
        
        features = {
            "shot_id": shot_id,
            
            # === ARM FEATURES ===
            "mean_right_elbow_angle": np.mean(right_elbow_angles),
            "max_right_elbow_angle": np.max(right_elbow_angles),
            "min_right_elbow_angle": np.min(right_elbow_angles),
            "range_right_elbow_angle": np.max(right_elbow_angles) - np.min(right_elbow_angles),
            "std_right_elbow_angle": np.std(right_elbow_angles),
            
            "mean_left_elbow_angle": np.mean(left_elbow_angles),
            "max_left_elbow_angle": np.max(left_elbow_angles),
            "min_left_elbow_angle": np.min(left_elbow_angles),
            "range_left_elbow_angle": np.max(left_elbow_angles) - np.min(left_elbow_angles),
            
            # Wrist features
            "max_right_wrist_height": np.min(right_wrist_heights),  # Min Y = highest point
            "max_left_wrist_height": np.min(left_wrist_heights),
            "right_wrist_velocity": calculate_velocity(right_wrist_heights),
            "left_wrist_velocity": calculate_velocity(left_wrist_heights),
            "right_wrist_acceleration": calculate_acceleration(right_wrist_heights),
            "right_wrist_path_length": right_wrist_path,
            "left_wrist_path_length": left_wrist_path,
            
            # === LEG FEATURES ===
            "mean_right_knee_angle": np.mean(right_knee_angles),
            "min_right_knee_angle": np.min(right_knee_angles),
            "max_right_knee_angle": np.max(right_knee_angles),
            "range_right_knee_angle": np.max(right_knee_angles) - np.min(right_knee_angles),
            
            "mean_left_knee_angle": np.mean(left_knee_angles),
            "min_left_knee_angle": np.min(left_knee_angles),
            "range_left_knee_angle": np.max(left_knee_angles) - np.min(left_knee_angles),
            
            # === HEAD FEATURES ===
            "mean_head_position_x": np.mean([p[0] for p in head_positions]),
            "mean_head_position_y": np.mean([p[1] for p in head_positions]),
            "head_movement_x": np.std([p[0] for p in head_positions]),
            "head_movement_y": np.std([p[1] for p in head_positions]),
            "mean_head_tilt": np.mean(head_tilts),
            "head_tilt_range": np.max(head_tilts) - np.min(head_tilts),
            
            # === HIP & TORSO FEATURES ===
            "mean_hip_rotation": np.mean(hip_rotations),
            "max_hip_rotation": np.max(hip_rotations),
            "min_hip_rotation": np.min(hip_rotations),
            "hip_rotation_range": np.max(hip_rotations) - np.min(hip_rotations),
            "std_hip_rotation": np.std(hip_rotations),
            
            "mean_shoulder_rotation": np.mean(shoulder_rotations),
            "shoulder_rotation_range": np.max(shoulder_rotations) - np.min(shoulder_rotations),
            
            "mean_torso_angle": np.mean(torso_angles),
            "max_torso_lean": np.max(np.abs(torso_angles)),
            "torso_angle_range": np.max(torso_angles) - np.min(torso_angles),
            "std_torso_angle": np.std(torso_angles),
            
            # === STANCE & STABILITY ===
            "mean_stance_width": np.mean(stance_widths),
            "max_stance_width": np.max(stance_widths),
            "min_stance_width": np.min(stance_widths),
            "stance_width_range": np.max(stance_widths) - np.min(stance_widths),
            "std_stance_width": np.std(stance_widths),
            
            # Center of mass / Balance
            "body_sway_x": sway_x,
            "body_sway_y": sway_y,
            "total_body_sway": np.sqrt(sway_x**2 + sway_y**2),
            "mean_base_of_support": np.mean(base_of_support),
            
            # Ankle stability
            "left_ankle_stability_x": left_ankle_movement[0],
            "left_ankle_stability_y": left_ankle_movement[1],
            "right_ankle_stability_x": right_ankle_movement[0],
            "right_ankle_stability_y": right_ankle_movement[1],
            
            # === SHOT CHARACTERISTICS ===
            "shot_duration_frames": len(df),
            "shot_duration_seconds": len(df) / 30,  # Assuming 30 FPS
            
            # Overall body coordination (difference between hip and shoulder rotation)
            "hip_shoulder_coordination": np.mean(np.abs(np.array(hip_rotations) - np.array(shoulder_rotations)))
        }
        
        return features
        
    except Exception as e:
        print(f"❌ Error processing {shot_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ---------------------------
# Main Processing Loop
# ---------------------------
# def main():
#     # Find all biomech CSV files
#     csv_files = list(Path(INPUT_FOLDER).glob("*_biomech.csv"))
    
#     if not csv_files:
#         print(f"❌ No *_biomech.csv files found in {INPUT_FOLDER}")
#         return
    
#     print(f"\n📊 Found {len(csv_files)} CSV file(s) to process")
#     print(f"Output folder: {OUTPUT_FOLDER}\n")
    
#     all_features = []
#     successful = 0
#     failed = 0
    
#     # Process each CSV
#     for idx, csv_path in enumerate(csv_files, 1):
#         shot_name = csv_path.stem.replace("_biomech", "")
#         print(f"[{idx}/{len(csv_files)}] Processing: {shot_name}")
        
#         features = extract_features_from_csv(csv_path, shot_name)
        
#         if features:
#             all_features.append(features)
            
#             # Save individual feature file
#             individual_output = os.path.join(OUTPUT_FOLDER, f"{shot_name}_features.csv")
#             pd.DataFrame([features]).to_csv(individual_output, index=False)
#             print(f"   ✓ Saved: {shot_name}_features.csv")
#             successful += 1
#         else:
#             failed += 1
    
#     # Save combined features file
#     if all_features:
#         combined_df = pd.DataFrame(all_features)
#         combined_path = os.path.join(OUTPUT_FOLDER, COMBINED_OUTPUT)
#         combined_df.to_csv(combined_path, index=False)
        
#         print(f"\n{'='*60}")
#         print(f"FEATURE EXTRACTION COMPLETE")
#         print(f"{'='*60}")
#         print(f"✅ Successful: {successful}")
#         print(f"❌ Failed: {failed}")
#         print(f"Total features extracted: {len(combined_df.columns) - 1}")  # Exclude shot_id
#         print(f"\n📁 Individual features saved to: {OUTPUT_FOLDER}")
#         print(f"📊 Combined features saved to: {combined_path}")
        
#         # Display feature categories
#         print(f"\n{'='*60}")
#         print("FEATURE CATEGORIES")
#         print(f"{'='*60}")
#         print("✓ Arm Features: Elbow angles, wrist heights, velocities, accelerations, path lengths")
#         print("✓ Leg Features: Knee angles, stance width")
#         print("✓ Head Features: Position, movement, tilt")
#         print("✓ Hip & Torso: Rotation, lean, coordination")
#         print("✓ Stability: Body sway, center of mass, ankle stability, base of support")
#         print("✓ Shot Characteristics: Duration, coordination metrics")
        
#     else:
#         print("\n❌ No features extracted from any file")

# if __name__ == "__main__":
#     main()

def main():
    # UPDATED: We now look for any CSV in the folder. 
    # We will filter them inside the loop to ensure they are pose data.
    all_files = list(Path(INPUT_FOLDER).glob("*.csv"))
    
    # Filter to only include original biomech and mirrored files
    csv_files = [f for f in all_files if f.name.endswith('_biomech.csv') or f.name.endswith('_mirrored.csv')]
    
    if not csv_files:
        print(f"❌ No suitable CSV files found in {INPUT_FOLDER}")
        return
    
    print(f"\n📊 Found {len(csv_files)} CSV file(s) to process")
    print(f"Output folder: {OUTPUT_FOLDER}\n")
    
    all_features = []
    successful = 0
    failed = 0
    
    # Process each CSV
    for idx, csv_path in enumerate(csv_files, 1):
        # UPDATED: Improved shot_name parsing to handle both suffixes
        shot_name = csv_path.stem.replace("_biomech", "").replace("_mirrored", "")
        
        # If it was a mirrored file, let's keep that in the ID so they stay unique
        current_id = csv_path.stem.replace("_biomech", "")
        
        print(f"[{idx}/{len(csv_files)}] Processing: {current_id}")
        
        # Pass the current_id (e.g., shot1_mirrored) so the shot_id column is correct
        features = extract_features_from_csv(csv_path, current_id)
        
        if features:
            all_features.append(features)
            
            # Save individual feature file
            individual_output = os.path.join(OUTPUT_FOLDER, f"{current_id}_features.csv")
            pd.DataFrame([features]).to_csv(individual_output, index=False)
            print(f"   ✓ Saved: {current_id}_features.csv")
            successful += 1
        else:
            failed += 1
    
    # Save combined features file
    if all_features:
        combined_df = pd.DataFrame(all_features)
        combined_path = os.path.join(OUTPUT_FOLDER, COMBINED_OUTPUT)
        combined_df.to_csv(combined_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"FEATURE EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful} (Originals + Mirrored)")
        print(f"❌ Failed: {failed}")
        print(f"Total features extracted: {len(combined_df.columns) - 1}")  # Exclude shot_id
        print(f"\n📁 Individual features saved to: {OUTPUT_FOLDER}")
        print(f"📊 Combined features saved to: {combined_path}")
        
        # Display feature categories
        print(f"\n{'='*60}")
        print("FEATURE CATEGORIES")
        print(f"{'='*60}")
        print("✓ Arm Features: Elbow angles, wrist heights, velocities, accelerations, path lengths")
        print("✓ Leg Features: Knee angles, stance width")
        print("✓ Head Features: Position, movement, tilt")
        print("✓ Hip & Torso: Rotation, lean, coordination")
        print("✓ Stability: Body sway, center of mass, ankle stability, base of support")
        print("✓ Shot Characteristics: Duration, coordination metrics")
        
    else:
        print("\n❌ No features extracted from any file")

if __name__ == "__main__":
    main()