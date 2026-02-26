"""
STEP 3 - RUN THIS TO VERIFY EVERYTHING WORKS
Checks that model, scaler, and sample data all work correctly.
Run: python ml/verify.py
"""

import pandas as pd
import numpy as np
import joblib
import os


def verify():
    print("=" * 50)
    print("VERIFYING PROJECT SETUP")
    print("=" * 50)

    errors = []

    # Check model.pkl
    if os.path.exists('ml/model.pkl'):
        print("✅ ml/model.pkl found")
    else:
        print("❌ ml/model.pkl NOT found - run python ml/train.py first")
        errors.append("model.pkl missing")

    # Check scaler.pkl
    if os.path.exists('ml/scaler.pkl'):
        print("✅ ml/scaler.pkl found")
    else:
        print("❌ ml/scaler.pkl NOT found - run python ml/train.py first")
        errors.append("scaler.pkl missing")

    # Check sample data
    if os.path.exists('data/sample_wednesday.csv'):
        df = pd.read_csv('data/sample_wednesday.csv')
        print(f"✅ data/sample_wednesday.csv found ({len(df)} rows, {len(df.columns)} columns)")
    else:
        print("❌ data/sample_wednesday.csv NOT found - run python ml/create_sample.py first")
        errors.append("sample_wednesday.csv missing")

    if errors:
        print("")
        print("Fix the errors above before running the app.")
        return

    # Try running a detection
    print("")
    print("Testing anomaly detection...")

    model = joblib.load('ml/model.pkl')
    meta = joblib.load('ml/scaler.pkl')
    scaler = meta['scaler']
    features = meta['features']

    df = pd.read_csv('data/sample_wednesday.csv', low_memory=False)
    df.columns = df.columns.str.strip()

    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)

    anomaly_count = (preds == -1).sum()
    print(f"✅ Detection works! Found {anomaly_count} anomalies out of {len(df)} flows")
    print("")
    print("=" * 50)
    print("ALL CHECKS PASSED! Run the app with:")
    print("  streamlit run app.py")
    print("=" * 50)


if __name__ == '__main__':
    verify()
