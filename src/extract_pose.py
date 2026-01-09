import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
VIDEO_FOLDER = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\raw_videos"
OUTPUT_FOLDER = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\pose_data"
MODEL_PATH = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\src\\models\\pose_landmarker_lite.task"

# Supported video formats
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------
# Pose Connections for Drawing
# ---------------------------
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

# ---------------------------
# Draw Landmarks Function
# ---------------------------
def draw_landmarks(image, landmarks):
    """Draw pose landmarks and connections on image"""
    h, w, _ = image.shape
    
    # Draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    # Draw landmarks
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(image, (cx, cy), 7, (255, 255, 255), 2)
    
    return image

# ---------------------------
# Process Single Video Function
# ---------------------------
def process_video(video_path, detector, output_folder):
    """Process a single video file"""
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Output paths
    output_csv = os.path.join(output_folder, f"{video_name}_biomech.csv")
    output_frame = os.path.join(output_folder, f"{video_name}_skeleton.jpg")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    
    data = []
    frame_id = 0
    saved_overlay = False
    detected_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # Convert to MediaPipe Image format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect pose
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks:
            pose_landmarks = detection_result.pose_landmarks[0]
            detected_frames += 1
            
            row = {"frame": frame_id, "timestamp": frame_id / fps}
            
            for idx, lm in enumerate(pose_landmarks):
                row[f"lm_{idx}_x"] = lm.x
                row[f"lm_{idx}_y"] = lm.y
                row[f"lm_{idx}_z"] = lm.z
                row[f"lm_{idx}_visibility"] = lm.visibility
            
            data.append(row)
            
            # Save one skeleton overlay frame (middle of video)
            if not saved_overlay and frame_id > total_frames // 2:
                annotated_frame = frame.copy()
                annotated_frame = draw_landmarks(annotated_frame, pose_landmarks)
                cv2.imwrite(output_frame, annotated_frame)
                saved_overlay = True
                print(f"✓ Saved overlay frame at frame {frame_id}")
        
        # Progress indicator
        if frame_id % 50 == 0:
            progress = (frame_id / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_id}/{total_frames} frames)")
    
    cap.release()
    
    # Save CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Success!")
        print(f"   CSV saved: {output_csv}")
        print(f"   Overlay saved: {output_frame}")
        print(f"   Frames processed: {frame_id}")
        print(f"   Pose detected in: {detected_frames} frames ({detected_frames/frame_id*100:.1f}%)")
        return True
    else:
        print(f"⚠️  Warning: No pose detected in any frame")
        return False

# ---------------------------
# Main Processing Loop
# ---------------------------
def main():
    # Get all video files
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(Path(VIDEO_FOLDER).glob(f"*{ext}"))
    
    if not video_files:
        print(f"❌ No video files found in {VIDEO_FOLDER}")
        print(f"Looking for extensions: {VIDEO_EXTENSIONS}")
        return
    
    print(f"\n🎬 Found {len(video_files)} video(s) to process")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # Initialize MediaPipe Pose Detector
    print(f"\n🔧 Initializing MediaPipe Pose Detector...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    detector = vision.PoseLandmarker.create_from_options(options)
    print("✓ Detector initialized")
    
    # Process each video
    successful = 0
    failed = 0
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}]")
        if process_video(str(video_path), detector, OUTPUT_FOLDER):
            successful += 1
        else:
            failed += 1
    
    detector.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output location: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()