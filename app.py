"""
CyberShield - SOC Anomaly Detection Dashboard
Main Streamlit application file.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from dotenv import load_dotenv

# Load .env file (only works locally - Streamlit Cloud uses Secrets instead)
load_dotenv()

# Import our custom modules
from ml.predict import load_model, detect_anomalies, get_top_features
from llm.explainer import explain_anomaly, generate_soc_summary

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the first streamlit command
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM STYLING (dark cybersecurity theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

    /* Main background */
    .stApp {
        background: #060d1a;
        color: #c9d8e8;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a1628;
        border-right: 1px solid #1a3050;
    }

    /* KPI metric cards */
    .kpi-card {
        background: linear-gradient(135deg, #0d1f38 0%, #0a1628 100%);
        border: 1px solid #1a3050;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        margin: 4px;
    }
    .kpi-value {
        font-size: 2.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #6a8caa;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Severity badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
    }
    .badge-critical { background: #ff2442; color: white; }
    .badge-high     { background: #ff6b35; color: white; }
    .badge-medium   { background: #ffaa00; color: #000; }
    .badge-low      { background: #00cc66; color: #000; }

    /* Alert cards */
    .alert-box {
        background: #0a1628;
        border: 1px solid #1a3050;
        border-left: 4px solid #ff2442;
        border-radius: 8px;
        padding: 18px;
        margin: 10px 0;
    }
    .alert-box.high   { border-left-color: #ff6b35; }
    .alert-box.medium { border-left-color: #ffaa00; }
    .alert-box.low    { border-left-color: #00cc66; }

    /* Section headers */
    h1 { color: #e0eaf5 !important; font-family: 'Inter', sans-serif !important; }
    h2, h3 { color: #c9d8e8 !important; }

    /* Divider */
    hr { border-color: #1a3050 !important; }

    /* Summary box */
    .summary-box {
        background: #0d1f38;
        border: 1px solid #1e4080;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 16px 0;
        line-height: 1.7;
    }

    /* Code / mono text */
    .mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #7ac0ff;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CyberShield")
    st.markdown("**Network Anomaly Detection**")
    st.markdown("*Powered by Isolation Forest + Gemini AI*")
    st.divider()

    # API Key input
    st.markdown("### 🔑 Gemini API Key")
    api_key_input = st.text_input(
        "Enter your key",
        type="password",
        help="Get free key at aistudio.google.com"
    )
    if api_key_input:
        os.environ['GEMINI_API_KEY'] = api_key_input
        st.success("API key set!")
    elif os.environ.get('GEMINI_API_KEY'):
        st.success("API key loaded from .env")
    else:
        st.warning("No API key - LLM features disabled")

    st.divider()

    # Data source
    st.markdown("### 📂 Data Source")
    data_mode = st.radio(
        "Choose:",
        ["Use sample data (demo)", "Upload Wednesday CSV"]
    )

    uploaded_file = None
    if data_mode == "Upload Wednesday CSV":
        uploaded_file = st.file_uploader(
            "Wednesday-workingHours CSV",
            type=['csv'],
            help="Upload Wednesday-workingHours.pcap_ISCX.csv"
        )

    st.divider()

    # Detection settings
    st.markdown("### ⚙️ Detection Settings")
    risk_threshold = st.slider(
        "Minimum risk score to show",
        min_value=0,
        max_value=100,
        value=30,
        help="Only show anomalies above this risk score"
    )
    max_llm_alerts = st.slider(
        "Max alerts to explain with AI",
        min_value=1,
        max_value=20,
        value=5,
        help="More = slower but more thorough"
    )

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#6a8caa; line-height:1.8'>
    <b>How it works:</b><br>
    🔵 <b>Isolation Forest</b> detects anomalies<br>
    🟣 <b>Gemini AI</b> explains WHY<br>
    ⚠️ LLM does NOT make predictions<br><br>
    <b>Dataset:</b> CICIDS2017<br>
    Monday = training (normal)<br>
    Wednesday = testing (attacks)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model...")
def get_model():
    try:
        model, scaler, features = load_model()
        return model, scaler, features, None
    except FileNotFoundError:
        return None, None, None, "Model files not found"
    except Exception as e:
        return None, None, None, str(e)


model, scaler, features, model_error = get_model()


# ─────────────────────────────────────────────
# LOAD DATA (cached per file)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data(source: str):
    """Load data from sample file or uploaded file"""
    if source == "sample":
        if os.path.exists('data/sample_wednesday.csv'):
            return pd.read_csv('data/sample_wednesday.csv', low_memory=False), None
        else:
            return None, "sample_wednesday.csv not found in data/ folder"
    else:
        return None, "Use uploaded_file directly"


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("# 🛡️ CyberShield SOC Dashboard")
st.markdown("Unsupervised ML Anomaly Detection  ·  Gemini AI Explanations  ·  CICIDS2017 Dataset")
st.divider()

# ─────────────────────────────────────────────
# SHOW MODEL ERROR IF ANY
# ─────────────────────────────────────────────
if model_error:
    st.error(f"⚠️ Model not loaded: {model_error}")
    st.info("""
    **To fix this, run these commands in order:**
    ```
    python ml/train.py
    python ml/create_sample.py
    python ml/verify.py
    ```
    Then refresh this page.
    """)
    st.stop()

# ─────────────────────────────────────────────
# LOAD THE DATA
# ─────────────────────────────────────────────
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file, low_memory=False)
    data_label = f"Uploaded: {uploaded_file.name}"
else:
    raw_df, load_err = load_data("sample")
    if load_err:
        st.error(f"⚠️ {load_err}")
        st.info("Run `python ml/create_sample.py` to create the sample file.")
        st.stop()
    data_label = "Demo: sample_wednesday.csv (2000 rows)"

st.caption(f"📊 Data source: {data_label}")

# ─────────────────────────────────────────────
# RUN ANOMALY DETECTION
# ─────────────────────────────────────────────
with st.spinner("Running Isolation Forest anomaly detection..."):
    result_df, used_features = detect_anomalies(raw_df, model, scaler, features)

# Filter by threshold
all_anomalies = result_df[result_df['is_anomaly']].copy()
filtered_anomalies = all_anomalies[all_anomalies['risk_score'] >= risk_threshold].sort_values(
    'risk_score', ascending=False
)

# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total_flows = len(result_df)
total_anomalies = len(filtered_anomalies)
anomaly_rate = (len(all_anomalies) / total_flows * 100) if total_flows > 0 else 0
avg_risk = filtered_anomalies['risk_score'].mean() if len(filtered_anomalies) > 0 else 0

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#4a9eff">{total_flows:,}</div>
        <div class="kpi-label">Total Flows Analyzed</div>
    </div>""", unsafe_allow_html=True)

with col2:
    color = "#ff2442" if total_anomalies > 50 else "#ff6b35" if total_anomalies > 10 else "#ffaa00"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color}">{total_anomalies}</div>
        <div class="kpi-label">Anomalies Detected</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#ffaa00">{anomaly_rate:.1f}%</div>
        <div class="kpi-label">Anomaly Rate</div>
    </div>""", unsafe_allow_html=True)

with col4:
    risk_color = "#ff2442" if avg_risk >= 70 else "#ff6b35" if avg_risk >= 40 else "#00cc66"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{risk_color}">{avg_risk:.0f}</div>
        <div class="kpi-label">Avg Risk Score /100</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### 📊 Risk Score Distribution")
    fig1 = px.histogram(
        result_df,
        x='risk_score',
        nbins=40,
        color_discrete_sequence=['#4a9eff'],
        labels={'risk_score': 'Risk Score', 'count': 'Number of Flows'}
    )
    fig1.add_vline(
        x=risk_threshold,
        line_dash="dash",
        line_color="#ff2442",
        annotation_text=f"Threshold ({risk_threshold})",
        annotation_font_color="#ff2442"
    )
    fig1.update_layout(
        plot_bgcolor='#0d1f38',
        paper_bgcolor='#0a1628',
        font_color='#c9d8e8',
        showlegend=False,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    st.markdown("### 🍩 Normal vs Anomalous Traffic")
    normal_count = len(result_df[~result_df['is_anomaly']])
    anomaly_count_chart = len(result_df[result_df['is_anomaly']])

    fig2 = go.Figure(data=[go.Pie(
        labels=['Normal Traffic', 'Anomalous Traffic'],
        values=[normal_count, anomaly_count_chart],
        hole=0.55,
        marker_colors=['#00cc66', '#ff2442'],
        textfont_size=13
    )])
    fig2.update_layout(
        plot_bgcolor='#0d1f38',
        paper_bgcolor='#0a1628',
        font_color='#c9d8e8',
        margin=dict(t=20, b=20),
        legend=dict(bgcolor='#0a1628')
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# ALERT TABLE
# ─────────────────────────────────────────────
st.divider()
st.markdown("### 🚨 Alert Queue")

if len(filtered_anomalies) == 0:
    st.success(f"✅ No anomalies detected above risk threshold of {risk_threshold}")
else:
    # Show nice table with selected columns
    display_cols = ['risk_score', 'anomaly_score']
    extra_cols = ['Flow Duration', 'Flow Bytes/s', 'Total Fwd Packets',
                  'Total Backward Packets', 'Label']
    for c in extra_cols:
        if c in filtered_anomalies.columns:
            display_cols.append(c)

    show_df = filtered_anomalies[display_cols].head(100).copy()
    show_df['risk_score'] = show_df['risk_score'].round(1)
    show_df['anomaly_score'] = show_df['anomaly_score'].round(4)

    st.dataframe(
        show_df.style.background_gradient(
            subset=['risk_score'],
            cmap='RdYlGn_r'
        ).format({'risk_score': '{:.1f}', 'anomaly_score': '{:.4f}'}),
        use_container_width=True,
        height=300
    )
    st.caption(f"Showing top {min(100, len(filtered_anomalies))} of {len(filtered_anomalies)} anomalies above threshold {risk_threshold}")


# ─────────────────────────────────────────────
# GEMINI AI ANALYSIS SECTION
# ─────────────────────────────────────────────
st.divider()
st.markdown("### 🤖 Gemini AI — SOC Alert Analysis")
st.markdown("""
> **How this works:** Isolation Forest detected the anomalies above.  
> Gemini AI now reads the feature values and explains *why* each one looks suspicious — like a senior analyst reviewing a junior's alerts.
""")

has_api_key = bool(os.environ.get('GEMINI_API_KEY'))

if not has_api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to enable AI explanations.")
elif len(filtered_anomalies) == 0:
    st.info("No anomalies to analyze.")
else:
    analyze_clicked = st.button(
        f"🔍 Analyze Top {min(max_llm_alerts, len(filtered_anomalies))} Alerts with Gemini",
        type="primary",
        use_container_width=False
    )

    if analyze_clicked:
        top_alerts = filtered_anomalies.head(max_llm_alerts)

        # ── Executive Summary ──────────────────
        st.markdown("#### 📋 SOC Executive Summary")
        with st.spinner("Gemini is writing executive summary..."):
            summary = generate_soc_summary(top_alerts)

        st.markdown(f"""
        <div class="summary-box">
            <b style='color:#4a9eff'>INCIDENT REPORT — AUTO-GENERATED</b><br><br>
            {summary}
        </div>""", unsafe_allow_html=True)

        # ── Individual Alert Cards ──────────────
        st.markdown("#### 🔎 Individual Alert Breakdown")
        st.caption("Each alert analyzed by Gemini AI based on network flow features")

        progress_bar = st.progress(0, text="Analyzing alerts...")

        for i, (idx, row) in enumerate(top_alerts.iterrows()):
            # Update progress bar
            progress_bar.progress(
                (i + 1) / len(top_alerts),
                text=f"Analyzing alert {i+1} of {len(top_alerts)}..."
            )

            # Get feature values to send to Gemini
            row_features = get_top_features(row, used_features, n=8)

            # Ask Gemini to explain
            with st.spinner(f"Gemini analyzing alert #{i+1}..."):
                analysis = explain_anomaly(row_features, row['risk_score'])

            # Get severity for styling
            severity = analysis.get('severity', 'Medium').lower()
            card_class = f"alert-box {severity}" if severity in ['high','medium','low'] else "alert-box"
            badge_class = f"badge-{severity}"

            # Display alert card
            with st.expander(
                f"🔴 Alert #{i+1}  |  Risk: {row['risk_score']:.0f}/100  |  {analysis.get('attack_type','Unknown')}  |  {analysis.get('severity','?')}",
                expanded=(i < 2)  # first 2 are open by default
            ):
                left_col, right_col = st.columns([3, 1])

                with left_col:
                    st.markdown(f"**🔍 Why it's suspicious:**")
                    st.markdown(f"> {analysis.get('explanation', 'N/A')}")

                    st.markdown(f"**⚠️ Key Indicator of Compromise:**")
                    st.code(analysis.get('ioc', 'N/A'), language=None)

                    st.markdown(f"**🛠️ Recommended Actions:**")
                    mitigation = analysis.get('mitigation', 'No steps available')
                    for line in mitigation.split('\n'):
                        line = line.strip()
                        if line:
                            st.markdown(f"  {line}")

                with right_col:
                    st.markdown(f"""
                    <div style='text-align:center; padding:10px'>
                        <span class='badge {badge_class}'>{analysis.get('severity','?').upper()}</span>
                        <br><br>
                        <div style='color:#6a8caa; font-size:0.8rem'>ATTACK TYPE</div>
                        <div style='font-size:0.95rem; font-weight:600; margin-top:4px'>{analysis.get('attack_type','Unknown')}</div>
                        <br>
                        <div style='color:#6a8caa; font-size:0.8rem'>RISK SCORE</div>
                        <div style='font-size:2rem; font-weight:700; font-family:JetBrains Mono; color:#ff6b35'>{row['risk_score']:.0f}</div>
                        <div style='color:#6a8caa; font-size:0.75rem'>out of 100</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Show the raw feature values that were sent to Gemini
                st.markdown("**🔢 Raw feature values sent to Gemini:**")
                st.json(row_features)

        progress_bar.empty()
        st.success(f"✅ Analysis complete! {len(top_alerts)} alerts analyzed by Gemini AI.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#3a5a7a; font-size:0.78rem; padding:10px'>
    CyberShield SOC Dashboard  ·  Isolation Forest (scikit-learn) + Gemini 1.5 Flash  ·  CICIDS2017 Dataset<br>
    <i>ML detects anomalies. LLM explains them. Humans decide.</i>
</div>
""", unsafe_allow_html=True)
