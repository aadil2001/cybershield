# 🛡️ CyberShield — SOC Anomaly Detection Dashboard

Unsupervised ML anomaly detection on CICIDS2017 network traffic, with Gemini AI explanations.

## Tech Stack
- **Anomaly Detection:** Isolation Forest (scikit-learn)
- **AI Explanations:** Google Gemini 1.5 Flash (free)
- **Dashboard:** Streamlit
- **Dataset:** CICIDS2017

---

## Setup Instructions (Windows)

### Step 1 — Create project folder
```
mkdir cybershield
cd cybershield
```

### Step 2 — Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```
pip install -r requirements.txt
```

### Step 4 — Add your data files
Place these inside the `data/` folder:
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`

### Step 5 — Add your Gemini API key
Edit `.env` file:
```
GEMINI_API_KEY=your_key_here
```
Get free key at: https://aistudio.google.com

### Step 6 — Train the model
```
python ml/train.py
```

### Step 7 — Create sample data
```
python ml/create_sample.py
```

### Step 8 — Verify everything works
```
python ml/verify.py
```

### Step 9 — Run the app
```
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Cloud

1. Create GitHub account and push these files (NOT the big CSVs)
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. In Secrets, add: `GEMINI_API_KEY = "your_key"`
5. Deploy!

Files to commit to GitHub:
- app.py
- requirements.txt
- ml/train.py, predict.py, create_sample.py, verify.py
- ml/model.pkl ✅
- ml/scaler.pkl ✅
- data/sample_wednesday.csv ✅ (small 2000-row file)
- llm/explainer.py
- .gitignore

Files NOT to commit (too large):
- data/Monday-WorkingHours.pcap_ISCX.csv ❌
- data/Wednesday-workingHours.pcap_ISCX.csv ❌
