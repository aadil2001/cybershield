"""
STEP 1 - RUN THIS FIRST
This file trains the Isolation Forest model on Monday (normal) traffic.
Run it once: python ml/train.py
It will create model.pkl and scaler.pkl inside the ml/ folder.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# These are the 21 features we use from the dataset
# Isolation Forest will learn what "normal" looks like from these
FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Total',
    'Bwd IAT Total',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward'
]


def train_model():
    monday_path = 'data/Monday-WorkingHours.pcap_ISCX.csv'

    # Check if file exists
    if not os.path.exists(monday_path):
        print(f"ERROR: Could not find {monday_path}")
        print("Please place Monday-WorkingHours.pcap_ISCX.csv inside the data/ folder")
        return

    print("Loading Monday (normal traffic) data...")
    print("This may take 20-30 seconds for large files...")
    df = pd.read_csv(monday_path, low_memory=False)

    # Clean column names (remove spaces from start/end)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Keep only the features we need
    available_features = [f for f in FEATURES if f in df.columns]
    print(f"Using {len(available_features)} features out of {len(FEATURES)} requested")

    if len(available_features) < 5:
        print("ERROR: Too few features found. Check your CSV column names.")
        print("Available columns:", list(df.columns[:20]))
        return

    X = df[available_features].copy()

    # Clean bad values (infinity, NaN)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    print(f"Training Isolation Forest on {len(X)} samples...")
    print("This will take about 1-2 minutes on your Ryzen 7600X...")

    # Scale features so all are on same range
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    # contamination=0.01 means we expect ~1% of even normal data to look weird
    model = IsolationForest(
        n_estimators=200,       # 200 trees - more = better but slower
        contamination=0.01,
        random_state=42,        # for reproducibility
        n_jobs=-1               # use all CPU cores (your 7600X has 12 threads)
    )
    model.fit(X_scaled)

    # Save model and scaler to disk
    joblib.dump(model, 'ml/model.pkl')
    joblib.dump({
        'scaler': scaler,
        'features': available_features
    }, 'ml/scaler.pkl')

    print("")
    print("SUCCESS! Files created:")
    print("  ml/model.pkl   - the trained Isolation Forest")
    print("  ml/scaler.pkl  - the feature scaler")
    print("")
    print("Now run: python ml/create_sample.py")


if __name__ == '__main__':
    train_model()
