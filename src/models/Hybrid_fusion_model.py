import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# --- 1. CONFIG & PATHS ---
SEQ_DIR = r"C:\Users\Dhivya Shankar\Desktop\BITSSem4\Project\data\sequences"
MASTER_CSV = r"C:\Users\Dhivya Shankar\Desktop\BITSSem4\Project\Manually_made_masterData.csv"

# Load the master file you just made
df = pd.read_csv(MASTER_CSV)
y_map = {'Good': 1, 'Bad': 0}

# Define Stats columns (Match Context)
stats_cols = ['bowler_type', 'innings', 'balls_remaining', 'runs_required_if_innings2', 
              'venue_expected_runrate_if_innings1', 'pressure_index_if_innings1', 
              'ground_scoring_bucket', 'pitch_type', 'weather_condition', 
              'wickets_left', 'match_format']

# --- 2. PREPROCESSING STATS ---
# We use a transformer to handle categorical (text) and numerical data separately
cat_cols = ['bowler_type', 'ground_scoring_bucket', 'pitch_type', 'weather_condition', 'match_format']
num_cols = [c for c in stats_cols if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Process the tabular data once
X_stats_all = preprocessor.fit_transform(df[stats_cols])

# --- 3. DATA ALIGNMENT (Matching Sequences with Stats) ---
X_seq_final, X_stats_final, y_final = [], [], []

for idx, row in df.iterrows():
    s_id = str(row['shot_id'])
    label = y_map.get(row['shot_quality'])
    
    # original + mirrored
    for suffix in ["", "_mirrored"]:
        path = os.path.join(SEQ_DIR, f"{s_id}{suffix}.npy")
        if os.path.exists(path):
            X_seq_final.append(np.load(path))
            X_stats_final.append(X_stats_all[idx])
            y_final.append(label)

X_seq_final = np.array(X_seq_final)
X_stats_final = np.array(X_stats_final)
y_final = np.array(y_final)

# --- 4. THE HYBRID MODEL DEFINITION ---
def build_hybrid_model(seq_shape, stats_shape):
    # Video Branch (LSTM)
    input_seq = Input(shape=seq_shape, name="Sequence_Input")
    x = layers.LSTM(64, return_sequences=False)(input_seq)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Stats Branch (Dense)
    input_stats = Input(shape=stats_shape, name="Stats_Input")
    y_branch = layers.Dense(32, activation='relu')(input_stats)
    y_branch = layers.Dropout(0.2)(y_branch)
    
    # CONCATENATION (The Fusion)
    combined = layers.concatenate([x, y_branch])
    
    # Joint Reasoning Layers
    z = layers.Dense(32, activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    output = layers.Dense(1, activation='sigmoid')(z)
    
    model = models.Model(inputs=[input_seq, input_stats], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 5. 5-FOLD CROSS VALIDATION ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_accs = []

print(f"Training Hybrid Fusion Model on {len(X_seq_final)} samples...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq_final, y_final), 1):
    model = build_hybrid_model((90, 28), (X_stats_final.shape[1],))
    
    model.fit(
        [X_seq_final[train_idx], X_stats_final[train_idx]], y_final[train_idx],
        epochs=40, batch_size=16, verbose=0
    )
    
    y_pred = (model.predict([X_seq_final[val_idx], X_stats_final[val_idx]]) > 0.5).astype(int)
    acc = accuracy_score(y_final[val_idx], y_pred)
    final_accs.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.2f}")

print(f"\nFINAL HYBRID ACCURACY: {np.mean(final_accs):.2f}")