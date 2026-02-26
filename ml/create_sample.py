"""
STEP 2 - RUN THIS AFTER TRAINING
This creates a small 2000-row sample from Wednesday data.
We need this small file to upload to Streamlit Cloud (big files cant be uploaded).
Run: python ml/create_sample.py
"""

import pandas as pd
import os


def create_sample():
    wednesday_path = 'data/Wednesday-workingHours.pcap_ISCX.csv'

    if not os.path.exists(wednesday_path):
        print(f"ERROR: Could not find {wednesday_path}")
        print("Please place Wednesday-workingHours.pcap_ISCX.csv inside the data/ folder")
        return

    print("Loading Wednesday data...")
    df = pd.read_csv(wednesday_path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows")

    # Take 2000 random rows - mix of normal and attack traffic
    sample = df.sample(n=2000, random_state=42)
    sample.to_csv('data/sample_wednesday.csv', index=False)

    print("")
    print("SUCCESS! Created: data/sample_wednesday.csv (2000 rows)")
    print("This small file will be uploaded to GitHub for Streamlit Cloud")
    print("")
    print("Next step: python ml/verify.py  (to check everything works)")


if __name__ == '__main__':
    create_sample()
