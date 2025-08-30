import os
from collections import defaultdict
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

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

    # Build prompt
    prompt = "We detected anomalies from aircraft blackbox data:\n"
    for s in summary_input:
        prompt += f"- {s['rule']}: {s['count']} times, from {s['first_occurrence']} to {s['last_occurrence']}, max={s['max_value']}\n"

    prompt += "\nPlease provide a structured investigation report including:\n"
    prompt += "1. WHY these anomalies likely happened\n"
    prompt += "2. HOW they contributed to the accident\n"
    prompt += "3. WHAT key moments led to loss of control\n"
    prompt += "4. SUGGESTIONS for preventing similar incidents in the future\n"

    # to Gemini
    response = model.generate_content(prompt)
    return response.text