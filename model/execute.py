import os
import sys
import importlib
import __main__
from datetime import datetime


# --- Ensure repo root is on sys.path so `model` package is importable ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))   # .../path/to/repo/model
REPO_ROOT = os.path.dirname(MODEL_DIR)                   # .../path/to/repo
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Import the real module as the package module ---
# This ensures the module is loaded as model.model_02 (package form)
model_mod = importlib.import_module("model.model_02")

# --- Make compatibility aliases for whatever name the pickle used ---
# If the pickle references top-level module 'model_02', map that name to the package module.
sys.modules["model.model_02"] = model_mod
sys.modules["model_02"] = model_mod

# --- If the pickle expected classes in __main__, expose them there too ---
for _name in (
    "FeatureEngineer",
    "AccidentRiskModel",
    "PhaseDetector",
    "RuleEngine",
    "Calibrator",
    "load_json",
    "safe_float",
):
    if hasattr(model_mod, _name):
        setattr(__main__, _name, getattr(model_mod, _name))

# --- Now import the names for local use (clean, package-style) ---
from model.model_02 import (
    AccidentRiskModel,
    FeatureEngineer,
    PhaseDetector,
    RuleEngine,
    Calibrator,
    load_json,
    safe_float,
)

import pandas as pd
import joblib
import json
import tensorflow as tf
from tensorflow import keras

# --------------------------
# Load artifacts (your original constants)
# --------------------------
ARTIFACT_DIR = "C:\\Users\\HP\\OneDrive\\Desktop\\Aircraft-data-monitoring-system\\model\\artifacts"
MODEL_PKL = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PKL = os.path.join(ARTIFACT_DIR, "scaler.pkl")
FEATURES_JSON = os.path.join(ARTIFACT_DIR, "features.json")
THRESHOLDS_JSON = os.path.join(ARTIFACT_DIR, "thresholds.json")
META_JSON = os.path.join(ARTIFACT_DIR, "meta.json")
AE_PATH = os.path.join(ARTIFACT_DIR, "autoencoder.h5")
ISO_PKL = os.path.join(ARTIFACT_DIR, "iso.pkl")
CALIBRATOR_PKL = os.path.join(ARTIFACT_DIR, "calibrator.pkl")
STD_PARAMS = os.path.join(ARTIFACT_DIR, "std_params.pkl")


def load_autoencoder_safely(path):
    try:
        # Try normal load
        return keras.models.load_model(path, compile=False)
    except Exception as e:
        print(" Standard load_model failed:", e)
        print(" Falling back to manual JSON+weights loader...")

        try:
            # Manual H5 extraction (for older Keras models)
            import h5py
            f = h5py.File(path, "r")
            if "model_config" in f.attrs:
                model_json = f.attrs["model_config"].decode("utf-8")
            else:
                raise ValueError("model_config not found inside H5")

            model = keras.models.model_from_json(model_json)
            model.load_weights(path)
            return model

        except Exception as e2:
            print(" Fallback also failed:", e2)
            raise RuntimeError("Could not load autoencoder model. Consider re-exporting in same Keras version.")




# --------------------------
# Load saved objects (unpickle happens here)
# --------------------------
scaler = joblib.load(SCALER_PKL)
fe: FeatureEngineer = joblib.load(MODEL_PKL)
feature_names = json.load(open(FEATURES_JSON))
thresholds = json.load(open(THRESHOLDS_JSON))
meta = json.load(open(META_JSON))
ae = load_autoencoder_safely(AE_PATH)
calibrator = joblib.load(CALIBRATOR_PKL)
iso = joblib.load(ISO_PKL)
std_params = joblib.load(STD_PARAMS)

phase_detector = PhaseDetector()
rule_engine = RuleEngine(thresholds)


def standardize_value(x, params):
    mad = params["mad"] if params["mad"] > 0 else 1.0
    return (x - params["median"]) / mad


# --------------------------
# Prediction function
# --------------------------
# def predict_single_row(row: pd.Series):
#     # Phase detection
#     phase, vs_fpm = phase_detector.detect(row.to_dict())

#     # Rule-based check
#     severity, troubles, alerts = rule_engine.check(phase, row.to_dict())

#     # Feature engineering
#     X = fe.transform(pd.DataFrame([row]))
#     Xs = scaler.transform(X)   

#     # Autoencoder anomaly score
#     recon = ae.predict(Xs, verbose=0)
#     ae_err = float(((Xs - recon) ** 2).mean())

#     # Danger Level (0–10)
#     danger_score = (severity * 0.4) + (ae_err * 8.0)   # weights tuned down
#     danger_level = round(min(10.0, danger_score), 2)

#     # Probabilities
#     # probs = {
#     #     "Normal": max(0.0, 1 - danger_level / 10.0),
#     #     "Minor": min(0.5, danger_level / 20.0),
#     #     "Major": min(0.6, danger_level / 12.0),
#     #     "Crash": min(0.9, danger_level / 15.0),
#     # }
#     # total = sum(probs.values())
#     # for k in probs:
#     #     probs[k] = round(probs[k] / total, 2)

#     combined = 0.5 * iso_score + 0.35 * ae_err + 0.15 * severity


#     # Calibrated crash probability

#     crash_prob = round(calibrator.predict_proba(combined), 3)


#     # Final formatted response
#     response = {
#         "Time": str(row.get("timestamp")),
#         "Phase": phase.title(),
#         "Danger Level": f"{danger_level}/10",
#         "Crash Probability": crash_prob,
#         "Trouble Params": troubles,
#         # "Alerts": alerts,
#     }
#     return response

def predict_single_row(row):
    # row is already a dict
    phase, _ = phase_detector.detect(row)
    severity, troubles, _ = rule_engine.check(phase, row)

    X = fe.transform(pd.DataFrame([row]))
    Xs = scaler.transform(X)

    iso_score = -iso.score_samples(Xs)[0]
    recon = ae.predict(Xs, verbose=0)
    ae_err = float(((Xs - recon) ** 2).mean())

    iso_score_std = standardize_value(iso_score, std_params["iso"])
    ae_err_std = standardize_value(ae_err, std_params["ae"])
    sev_std = standardize_value(severity, std_params["sev"])

    combined = 0.5 * iso_score_std + 0.35 * ae_err_std + 0.15 * sev_std

    crash_prob = round(calibrator.predict_proba(combined), 3)
    danger_level = round(crash_prob * 10, 2)

    if crash_prob > 0.8 and not troubles:
        troubles.append("Unusual data pattern detected (anomaly)")

    return {
        "Time": str(row.get("timestamp")),
        "Phase": phase.title(),
        "Danger Level": f"{danger_level}/10",
        "Crash Probability": crash_prob,
        "Trouble Params": troubles,
    }



def convert_timestamp_to_float(payload: dict):
   # convert timestamp into float . Drops timestamp if error
    data = payload.copy()  # avoid mutating original input
    
    ts = data.get("timestamp")
    if ts:
        try:
            # TS format: "YYYY-MM-DD HH:MM:SS"
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            data["timestamp"] = dt.timestamp()  
        except ValueError:
            print(f"Warning: Invalid timestamp format: {ts}, dropping field.")
            data.pop("timestamp", None)
    
    return data




def predict(datarow):
    try:
        
        newRow = convert_timestamp_to_float(datarow)
        res = predict_single_row(newRow)
        print(json.dumps(res, indent=2))
        return json.dumps(res, indent=2)

    except Exception as e:
        print("Error occured in predict function", str(e))


