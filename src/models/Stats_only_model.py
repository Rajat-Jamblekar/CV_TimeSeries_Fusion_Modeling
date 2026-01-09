import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# 1. Load Data
df = pd.read_csv(r'C:\Users\Dhivya Shankar\Desktop\BITSSem4\Project\src\transformed_cricket_shots.csv')

# 2. SELECT ONLY STATS COLUMNS (The "Control" Group)
# We strictly exclude biomechanics (angles, velocities, etc.)
stats_features = [
    #'shot_type',
    'bowler_type',
    'innings', 
    'balls_remaining',
    'runs_required_if_innings2',
    'venue_expected_runrate_if_innings1',
    'pressure_index_if_innings1',
    'ground_scoring_bucket',
    'pitch_type',
    'weather_condition',
    'wickets_left',
    'match_format'
]

target_col = 'shot_quality' # or 'shot_quality_label' if you have 0/1

# Check if these columns actually exist in your CSV
existing_cols = [c for c in stats_features if c in df.columns]
print(f"Using these STATS columns: {existing_cols}")

X = df[existing_cols]

# Map Target to 0/1 if strictly needed (XGBoost often handles strings, but safer to map)
y = df[target_col].map({'Good': 1, 'Bad': 0, 'good': 1, 'bad': 0})

# 3. Preprocessing
# Stats are mostly categorical (Text), so we need OneHotEncoding
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# 4. Initialize Stats-Only Model
# We use a smaller tree depth because stats data is simple
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        max_depth=3,  
        n_estimators=50 
    ))
])

# 5. Evaluate (Stratified K-Fold)
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\n--------------------------------------")
print("RESULTS: MODEL 1 (STATS-ONLY BASELINE)")
print("--------------------------------------")
print(f"Average Accuracy: {acc_scores.mean():.2f}")
print(f"Average F1-Score: {f1_scores.mean():.2f}")
print("--------------------------------------")

# Interpretation Help
if acc_scores.mean() < 0.65:
    print(">> SUCCESS! The accuracy is low, proving that context alone is not enough.")
else:
    print(">> NOTE: Accuracy is high. Maybe your 'Shot Type' (e.g. Drive) is highly correlated with 'Good'.")

# Add this at the bottom of your script or run in a new cell
print("\n--- DIAGNOSIS: WHY IS ACCURACY SO HIGH? ---")
print("1. Class Balance (How many Good vs Bad?):")
# print(df['shot_quality'].value_counts(normalize=True))

print("\n2. Correlation Check (Does Shot Type give away the answer?):")
# See if 'shot_type' is perfectly predicting 'shot_quality'
ct = pd.crosstab(df['shot_type'], df['shot_quality'])
print(ct)