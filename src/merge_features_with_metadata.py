import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
FEATURES_CSV = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\features\\all_shots_features.csv"
METADATA_CSV = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\metadata.csv"
OUTPUT_CSV = "features_master.csv"

# ---------------------------
# Load CSVs
# ---------------------------
features_df = pd.read_csv(FEATURES_CSV)
metadata_df = pd.read_csv(METADATA_CSV)

# ---------------------------
# Normalize shot_id
# ---------------------------
features_df["shot_id"] = features_df["shot_id"].str.lower().str.strip()
metadata_df["shot_id"] = metadata_df["shot_id"].str.lower().str.strip()

# ---------------------------
# Normalize shot_quality
# ---------------------------
metadata_df["shot_quality"] = metadata_df["shot_quality"].str.lower().str.strip()

# ---------------------------
# Merge features + metadata
# ---------------------------
merged_df = pd.merge(
    features_df,
    metadata_df,
    on="shot_id",
    how="inner"
)

# ---------------------------
# Encode target variable
# ---------------------------
merged_df["shot_quality_label"] = merged_df["shot_quality"].map({
    "good": 1,
    "bad": 0
})

# ---------------------------
# Encode bowler type
# ---------------------------
merged_df["bowler_type_encoded"] = merged_df["bowler_type"].astype("category").cat.codes

# ---------------------------
# Save final dataset
# ---------------------------
merged_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Merge successful")
print(f"Final dataset saved as: {OUTPUT_CSV}")
print("\nDataset shape:", merged_df.shape)
print("\nSample rows:")
print(merged_df[[
    "shot_id",
    "shot_quality",
    "shot_quality_label",
    "bowler_type",
    "bowler_type_encoded"
]].head())
