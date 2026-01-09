import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

SEQ_DIR = "C:\\Users\\Dhivya Shankar\\Desktop\\BITSSem4\\Project\\data\\sequences"
# Note: You'll need to update your master CSV to include the 'mirrored' shot IDs and their labels
MASTER_CSV = r"C:\Users\Dhivya Shankar\Desktop\BITSSem4\Project\data\features\all_shots_features.csv"

# --- DATA PREP ---
df = pd.read_csv(MASTER_CSV)
# Map Good=1, Bad=0
y_map = {'Good': 1, 'Bad': 0}
X, y = [], []

for idx, row in df.iterrows():
    path = os.path.join(SEQ_DIR, f"{row['shot_id']}.npy")
    mirrored_path = os.path.join(SEQ_DIR, f"{row['shot_id']}_mirrored.npy")
    
    # Original Shot
    if os.path.exists(path):
        X.append(np.load(path))
        y.append(y_map[row['shot_quality']])
    
    # Mirrored Shot (Inherits same label)
    if os.path.exists(mirrored_path):
        X.append(np.load(mirrored_path))
        y.append(y_map[row['shot_quality']])

X, y = np.array(X), np.array(y)

# --- MODEL ---
def build_lstm():
    model = Sequential([
        LSTM(64, input_shape=(90, 28), return_sequences=False),
        BatchNormalization(),
        Dropout(0.5), # Strong dropout for small dataset
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- EVALUATION ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in skf.split(X, y):
    m = build_lstm()
    m.fit(X[train_idx], y[train_idx], epochs=40, batch_size=16, verbose=0)
    p = (m.predict(X[val_idx]) > 0.5).astype(int)
    scores.append(accuracy_score(y[val_idx], p))

print(f"Video-Only LSTM Accuracy: {np.mean(scores):.2f}")