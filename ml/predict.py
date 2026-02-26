"""
This file is used by app.py to run anomaly detection.
You do NOT need to run this directly.
"""

import pandas as pd
import numpy as np
import joblib


def load_model():
    """Load the trained model and scaler from disk"""
    model = joblib.load('ml/model.pkl')
    meta = joblib.load('ml/scaler.pkl')
    return model, meta['scaler'], meta['features']


def detect_anomalies(df, model, scaler, features):
    """
    Run Isolation Forest on the dataframe.
    Returns the same dataframe with 3 new columns added:
      - is_anomaly: True if flagged as attack
      - anomaly_score: raw model score (lower = more suspicious)
      - risk_score: 0-100 scale (higher = more suspicious, easier to read)
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Use only features the model was trained on
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    X_scaled = scaler.transform(X)

    # -1 = anomaly, 1 = normal
    predictions = model.predict(X_scaled)

    # Lower score = more anomalous
    scores = model.score_samples(X_scaled)

    df['is_anomaly'] = predictions == -1
    df['anomaly_score'] = scores

    # Convert to 0-100 risk scale (higher = riskier)
    min_s = scores.min()
    max_s = scores.max()
    df['risk_score'] = ((scores - max_s) / (min_s - max_s + 1e-9) * 100).clip(0, 100)

    return df, available


def get_top_features(row, features, n=8):
    """
    Extract top N feature values from a row to send to Gemini.
    Returns a clean dict with feature name -> value.
    """
    result = {}
    for f in features[:n]:
        if f in row.index:
            val = row[f]
            if pd.notna(val) and not np.isinf(val):
                result[f] = round(float(val), 4)
    return result
