"""
This file handles all communication with Google Gemini AI.
Gemini is used ONLY for explanation - NOT for detecting attacks.
Detection is done by Isolation Forest in ml/predict.py
"""

import google.generativeai as genai
import json
import os
import time


def init_gemini():
    """Initialize Gemini with API key from environment"""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
    genai.configure(api_key=api_key)
    # gemini-1.5-flash-8b = most generous free tier: 1000 requests/day, 15/min
    return genai.GenerativeModel('gemini-2.5-flash-lite')


def call_gemini_with_retry(model, prompt, max_retries=3):
    """
    Call Gemini with automatic retry if rate limited.
    Waits the exact number of seconds Gemini tells us to wait.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            error_msg = str(e)

            # Check if it's a rate limit error (429)
            if '429' in error_msg:
                # Try to extract how many seconds to wait from the error
                wait_seconds = 15  # default wait
                if 'retry_delay' in error_msg:
                    import re
                    match = re.search(r'seconds: (\d+)', error_msg)
                    if match:
                        wait_seconds = int(match.group(1)) + 2  # add 2s buffer

                if attempt < max_retries - 1:
                    print(f"Rate limited. Waiting {wait_seconds}s before retry {attempt+2}/{max_retries}...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    raise Exception(f"Rate limit hit after {max_retries} retries. Try again in a few minutes.")
            else:
                raise  # Not a rate limit error, raise immediately

    raise Exception("Max retries exceeded")


def explain_anomaly(row_features: dict, risk_score: float) -> dict:
    """
    Send anomaly feature data to Gemini.
    Returns a dict with: attack_type, severity, explanation, ioc, mitigation
    """
    raw = ""
    try:
        model = init_gemini()
        features_str = json.dumps(row_features, indent=2)

        prompt = f"""You are a senior SOC (Security Operations Center) analyst reviewing network traffic.
Our ML model (Isolation Forest) flagged this network flow as anomalous.

ANOMALY DATA:
- Risk Score: {risk_score:.1f} out of 100 (higher means more suspicious)
- Network Flow Measurements:
{features_str}

Your job is to explain WHY this is suspicious to a junior analyst.

Reply ONLY with this exact JSON structure. No extra text. No markdown. No code blocks:
{{
  "attack_type": "one of: DoS Attack, Port Scan, Brute Force, Data Exfiltration, Web Attack, Botnet, Infiltration, Unknown",
  "severity": "one of: Critical, High, Medium, Low",
  "explanation": "2-3 sentences explaining specifically which feature values look suspicious and why",
  "ioc": "the single most important indicator of compromise you see in this data",
  "mitigation": "3 specific numbered steps a SOC analyst should take right now"
}}"""

        raw = call_gemini_with_retry(model, prompt)

        # Gemini sometimes wraps in ```json ... ``` - remove that
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "attack_type": "Unknown",
            "severity": "Medium",
            "explanation": "Could not parse Gemini response. Raw output: " + raw[:200],
            "ioc": "Manual review required",
            "mitigation": "1. Review raw flow data\n2. Check firewall logs\n3. Escalate to senior analyst"
        }
    except Exception as e:
        return {
            "attack_type": "Error",
            "severity": "Low",
            "explanation": f"Gemini API error: {str(e)}",
            "ioc": "N/A",
            "mitigation": "Check your GEMINI_API_KEY in .env file or wait a few minutes and retry."
        }


def generate_soc_summary(anomalies_df) -> str:
    """
    Generate an executive summary of all detected anomalies.
    """
    try:
        model = init_gemini()

        total = len(anomalies_df)
        avg_risk = anomalies_df['risk_score'].mean()
        high_risk = len(anomalies_df[anomalies_df['risk_score'] >= 70])
        medium_risk = len(anomalies_df[(anomalies_df['risk_score'] >= 40) & (anomalies_df['risk_score'] < 70)])

        prompt = f"""You are a SOC team lead writing a brief incident summary report.

Detection Results:
- Total anomalies flagged: {total}
- High risk (score 70-100): {high_risk}
- Medium risk (score 40-69): {medium_risk}
- Average risk score: {avg_risk:.1f}/100

Write exactly 3 sentences in professional SOC language summarizing:
1. What was detected
2. The severity level
3. Recommended immediate action

Keep it concise and professional."""

        return call_gemini_with_retry(model, prompt)

    except Exception as e:
        return f"Summary unavailable: {str(e)}"
