import os
import time
from collections import defaultdict
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def summarize_anomalies(anomalies):
    grouped = defaultdict(list)

    for anomaly in anomalies:
        rule = anomaly["rule"]
        grouped[rule].append(anomaly)

    summary_input = []

    for rule, events in grouped.items():
        timestamps = [datetime.fromisoformat(e["timestamp"].replace("Z","")) for e in events]
        values = [e["value"] for e in events if isinstance(e["value"], (int,float))]
        summary_input.append({
            "rule": rule,
            "count": len(events),
            "first_occurrence": min(timestamps).isoformat(),
            "last_occurrence": max(timestamps).isoformat(),
            "max_value": max(values) if values else None
        })

    return summary_input


def generate_report(anomalies):

    # Summarise anomalies 
    summary_input = summarize_anomalies(anomalies)
    #print("Summary" , summary_input)

    if not summary_input:
        return "This is a smooth flight. No anomalies detected."

    # Build prompt
    prompt = "You are an aviation accident investigator. "
    prompt += "Write a structured **Accident Investigation Report** based ONLY on the anomalies provided below. "
    prompt += "Do not include disclaimers, prefaces, or notes about limited data. "
    prompt += "Begin directly with the report.\n\n"

    prompt += "### Anomaly Summary\n"
    for s in summary_input:
        prompt += f"- {s['rule']}: {s['count']} times, from {s['first_occurrence']} to {s['last_occurrence']}, max={s['max_value']}\n"

    prompt += "\n### Report Format\n"
    prompt += "## Accident Investigation Report\n"
    prompt += "### 1. Cause Analysis (WHY)\n"
    prompt += "### 2. Contribution to Accident (HOW)\n"
    prompt += "### 3. Key Moments Leading to Loss of Control (WHAT)\n"
    prompt += "### 4. Preventive Suggestions\n"

    # to Gemini
    # Retry logic (up to 2 attempts)
    for attempt in range(2): 
        try:
            response = model.generate_content(prompt)
            return response.text
        except ResourceExhausted as e:
            wait_time = 30 * (attempt + 1)
            print(f"Rate limit hit. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")

    raise RuntimeError("Failed to generate report after retries.")