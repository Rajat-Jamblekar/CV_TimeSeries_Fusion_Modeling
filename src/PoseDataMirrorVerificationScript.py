import pandas as pd
import matplotlib.pyplot as plt

def verify_mirroring(original_csv, mirrored_csv, frame_idx=0):
    df_orig = pd.read_csv(original_csv)
    df_mirr = pd.read_csv(mirrored_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def plot_skeleton(df, ax, title):
        # Extract X and Y for all landmarks in the specified frame
        x_cols = [c for c in df.columns if '_x' in c]
        y_cols = [c for c in df.columns if '_y' in c]
        
        x_vals = df.iloc[frame_idx][x_cols].values
        y_vals = df.iloc[frame_idx][y_cols].values

        # Scatter plot for joints
        ax.scatter(x_vals, y_vals, s=20, c='red')
        
        # Draw some connections (e.g., Shoulders 11-12, Hips 23-24)
        # Suffixes are 'visibility' based on your data
        connections = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15)]
        for start, end in connections:
            try:
                ax.plot([df.at[frame_idx, f'lm_{start}_x'], df.at[frame_idx, f'lm_{end}_x']],
                        [df.at[frame_idx, f'lm_{start}_y'], df.at[frame_idx, f'lm_{end}_y']], 'blue')
            except KeyError:
                continue

        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0) # Invert Y because image coordinates start at top
        ax.set_aspect('equal')

    plot_skeleton(df_orig, ax1, "Original Pose")
    plot_skeleton(df_mirr, ax2, "Mirrored Pose (Augmented)")

    plt.tight_layout()
    plt.show()

# Test it with one of your files
orig = r"C:\Users\Dhivya Shankar\Desktop\BITSSem4\Project\data\pose_data\shot1_biomech.csv"
mirr = r"C:\Users\Dhivya Shankar\Desktop\BITSSem4\Project\data\pose_data\shot1_biomech_mirrored.csv"
verify_mirroring(orig, mirr)