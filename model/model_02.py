

import os
import sys
import json
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --------------------------
# Artifact paths
# --------------------------
ARTIFACT_DIR = "C:\\Users\\HP\\OneDrive\\Desktop\\Aircraft-data-monitoring-system\\model\\artifacts"
MODEL_PKL = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PKL = os.path.join(ARTIFACT_DIR, "scaler.pkl")
FEATURES_JSON = os.path.join(ARTIFACT_DIR, "features.json")
THRESHOLDS_JSON = os.path.join(ARTIFACT_DIR, "thresholds.json")
META_JSON = os.path.join(ARTIFACT_DIR, "meta.json")
AE_PATH = os.path.join(ARTIFACT_DIR, "autoencoder.h5")




# --------------------------
# Raw columns
# --------------------------
RAW_COLUMNS = [
    "timestamp",
    "altitude",
    "engine1_rpm", "engine2_rpm",
    "engine1_temp_C", "engine2_temp_C",
    "vibration_mm_s",
    "fuel_flow_kg_h",
    "oil_pressure_psi",
    "heading_deg",
    "hydraulic_pressure_psi",
    "electrical_load_amp",
    "cabin_pressure_ft",
    "angle_of_attack_deg",
    "flaps_deg",
    "spoiler_percent",
]

def sanitize_columns(cols: List[str]) -> List[str]:
    return [c.strip().replace(" ", "_") for c in cols]

SANITIZED_COLUMNS = sanitize_columns(RAW_COLUMNS)

PHASES = ["ground_takeoff", "climb", "cruise", "descent_landing"]

# DEFAULT_THRESHOLDS = {
#     "ground_takeoff": {"engine_rpm": (500, 10000), "engine_temp_C": (50, 900), "vibration_mm_s": (0.0, 5.0),
#                        "fuel_flow_kg_h": (100, 6000), "oil_pressure_psi": (20, 100), "hydraulic_pressure_psi": (2000, 3500),
#                        "electrical_load_amp": (0, 400), "cabin_pressure_ft": (0, 3000), "angle_of_attack_deg": (0, 15),
#                        "flaps_deg": (5, 20), "spoiler_percent": (0, 5)},
#     "climb": {"engine_rpm": (1000, 11000), "engine_temp_C": (100, 950), "vibration_mm_s": (0.0, 4.0),
#               "fuel_flow_kg_h": (200, 5000), "oil_pressure_psi": (30, 110), "hydraulic_pressure_psi": (2000, 3500),
#               "electrical_load_amp": (0, 400), "cabin_pressure_ft": (0, 8000), "angle_of_attack_deg": (2, 12),
#               "flaps_deg": (0, 5), "spoiler_percent": (0, 5)},
#     "cruise": {"engine_rpm": (800, 12000), "engine_temp_C": (150, 950), "vibration_mm_s": (0.0, 3.5),
#                "fuel_flow_kg_h": (100, 4000), "oil_pressure_psi": (30, 110), "hydraulic_pressure_psi": (2000, 3500),
#                "electrical_load_amp": (0, 400), "cabin_pressure_ft": (0, 10000), "angle_of_attack_deg": (0, 12),
#                "flaps_deg": (0, 1), "spoiler_percent": (0, 2)},
#     "descent_landing": {"engine_rpm": (300, 9000), "engine_temp_C": (80, 900), "vibration_mm_s": (0.0, 4.5),
#                         "fuel_flow_kg_h": (50, 3500), "oil_pressure_psi": (25, 110), "hydraulic_pressure_psi": (2000, 3500),
#                         "electrical_load_amp": (0, 400), "cabin_pressure_ft": (0, 8000), "angle_of_attack_deg": (0, 14),
#                         "flaps_deg": (5, 30), "spoiler_percent": (0, 60)},
# }

DEFAULT_THRESHOLDS = {
    "ground_takeoff": {
        "altitude": (0, 1500),  # ft AGL
        "engine1_rpm": (500, 10200),
        "engine2_rpm": (500, 10200),
        "engine1_temp_C": (50, 880),
        "engine2_temp_C": (50, 880),
        "vibration_mm_s": (0.0, 5.0),
        "fuel_flow_kg_h": (100, 6000),
        "oil_pressure_psi": (20, 100),
        "heading_deg": (0, 360),  # unrestricted
        "hydraulic_pressure_psi": (2500, 3200),
        "electrical_load_amp": (0, 400),
        "cabin_pressure_ft": (0, 3000),
        "angle_of_attack_deg": (0, 15),
        "flaps_deg": (5, 20),
        "spoiler_percent": (0, 5)
    },
    "climb": {
        "altitude": (1500, 35000),
        "engine1_rpm": (1000, 10500),
        "engine2_rpm": (1000, 10500),
        "engine1_temp_C": (100, 920),
        "engine2_temp_C": (100, 920),
        "vibration_mm_s": (0.0, 4.0),
        "fuel_flow_kg_h": (200, 5000),
        "oil_pressure_psi": (30, 110),
        "heading_deg": (0, 360),
        "hydraulic_pressure_psi": (2500, 3200),
        "electrical_load_amp": (0, 400),
        "cabin_pressure_ft": (0, 8000),
        "angle_of_attack_deg": (2, 12),
        "flaps_deg": (0, 5),
        "spoiler_percent": (0, 5)
    },
    "cruise": {
        "altitude": (28000, 41000),
        "engine1_rpm": (800, 9800),
        "engine2_rpm": (800, 9800),
        "engine1_temp_C": (150, 900),
        "engine2_temp_C": (150, 900),
        "vibration_mm_s": (0.0, 3.5),
        "fuel_flow_kg_h": (100, 4000),
        "oil_pressure_psi": (30, 110),
        "heading_deg": (0, 360),
        "hydraulic_pressure_psi": (2500, 3200),
        "electrical_load_amp": (0, 400),
        "cabin_pressure_ft": (0, 10000),
        "angle_of_attack_deg": (0, 8),
        "flaps_deg": (0, 1),
        "spoiler_percent": (0, 2)
    },
    "descent_landing": {
        "altitude": (0, 15000),
        "engine1_rpm": (300, 9000),
        "engine2_rpm": (300, 9000),
        "engine1_temp_C": (80, 880),
        "engine2_temp_C": (80, 880),
        "vibration_mm_s": (0.0, 4.5),
        "fuel_flow_kg_h": (50, 3500),
        "oil_pressure_psi": (25, 110),
        "heading_deg": (0, 360),
        "hydraulic_pressure_psi": (2500, 3200),
        "electrical_load_amp": (0, 400),
        "cabin_pressure_ft": (0, 8000),
        "angle_of_attack_deg": (0, 14),
        "flaps_deg": (5, 30),
        "spoiler_percent": (0, 60)
    }
}

def ensure_artifacts_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str):
    ensure_artifacts_dir()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Dict[str, Any]:     
    with open(path, "r") as f:
        return json.load(f)

# --------------------------
# Utilities
# --------------------------
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - mu)) + 1e-6
    return (x - mu) / (1.4826 * mad)

def parse_ts(ts: Any) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(ts)
    except Exception:
        return None

# --------------------------
# Runtime / Phase detection
# --------------------------
@dataclass
class RuntimeState:
    last_timestamp: Optional[pd.Timestamp] = None
    last_altitude: Optional[float] = None
    phase: str = "ground_takeoff"

    def update(self, timestamp: Optional[pd.Timestamp], altitude: Optional[float]) -> float:
        vs_fpm = 0.0
        if timestamp is not None and altitude is not None and self.last_timestamp is not None and self.last_altitude is not None:
            dt = (timestamp - self.last_timestamp).total_seconds()
            if dt > 0:
                vs_fpm = (altitude - self.last_altitude) * 60.0 / dt
        if timestamp is not None:
            self.last_timestamp = timestamp
        if altitude is not None:
            self.last_altitude = altitude
        return vs_fpm

class PhaseDetector:
    def __init__(self):
        self.state = RuntimeState()

    def detect(self, row: Dict[str, Any]) -> Tuple[str, float]:
        ts = parse_ts(row.get("timestamp"))
        alt = safe_float(row.get("altitude"))
        flaps = safe_float(row.get("flaps_deg"))
        spoiler = safe_float(row.get("spoiler_percent"))
        aoa = safe_float(row.get("angle_of_attack_deg"))
        vs_fpm = self.state.update(ts, alt)

        altitude = alt if alt is not None else 0.0
        flaps = flaps if flaps is not None else 0.0
        spoiler = spoiler if spoiler is not None else 0.0
        aoa = aoa if aoa is not None else 0.0

        phase = "cruise"
        if altitude < 1500 and vs_fpm > 800 and flaps >= 5 and aoa >= 5:
            phase = "ground_takeoff"
        elif vs_fpm > 500 and flaps <= 5 and altitude >= 1500:
            phase = "climb"
        elif abs(vs_fpm) <= 200 and altitude >= 10000 and flaps <= 1 and spoiler <= 2:
            phase = "cruise"
        elif vs_fpm < -500:
            phase = "descent_landing"
        else:
            if altitude < 1500:
                phase = "ground_takeoff" if vs_fpm >= 0 else "descent_landing"
            elif altitude < 10000 and vs_fpm > 0:
                phase = "climb"
            elif altitude >= 10000:
                phase = "cruise"

        self.state.phase = phase
        return phase, vs_fpm

# --------------------------
# Feature Engineering
# --------------------------
class FeatureEngineer:
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names_ = feature_names

    def fit(self, df: pd.DataFrame):
        df = self._sanitize_df(df)
        feats, _ = self._build_features(df)
        self.feature_names_ = list(feats.columns)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._sanitize_df(df)
        feats, _ = self._build_features(df)
        feats = feats.reindex(columns=self.feature_names_, fill_value=0.0)
        return feats.to_numpy(dtype=np.float32)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = sanitize_columns(list(df.columns))
        for raw, sani in zip(RAW_COLUMNS, SANITIZED_COLUMNS):
            if sani not in df.columns:
                df[sani] = np.nan
        if "timestamp" in df.columns:
            df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["timestamp_parsed"] = pd.NaT
        return df

    def _build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.sort_values("timestamp_parsed")
        alt = pd.to_numeric(df["altitude"], errors="coerce")
        ts = df["timestamp_parsed"]
        dt = ts.diff().dt.total_seconds().fillna(1.0).replace(0, 1.0)
        vs_fpm = (alt.diff().fillna(0.0) * 60.0 / dt).clip(-6000, 6000)
        hd = pd.to_numeric(df["heading_deg"], errors="coerce")
        hd_rad = np.deg2rad(hd.fillna(0.0))
        feats = pd.DataFrame({
            "altitude": alt.fillna(0.0),
            "engine1_rpm": pd.to_numeric(df["engine1_rpm"], errors="coerce").fillna(0.0),
            "engine2_rpm": pd.to_numeric(df["engine2_rpm"], errors="coerce").fillna(0.0),
            "engine1_temp_C": pd.to_numeric(df["engine1_temp_C"], errors="coerce").fillna(0.0),
            "engine2_temp_C": pd.to_numeric(df["engine2_temp_C"], errors="coerce").fillna(0.0),
            "vibration_mm_s": pd.to_numeric(df["vibration_mm_s"], errors="coerce").fillna(0.0),
            "fuel_flow_kg_h": pd.to_numeric(df["fuel_flow_kg_h"], errors="coerce").fillna(0.0),
            "oil_pressure_psi": pd.to_numeric(df["oil_pressure_psi"], errors="coerce").fillna(0.0),
            "hydraulic_pressure_psi": pd.to_numeric(df["hydraulic_pressure_psi"], errors="coerce").fillna(0.0),
            "electrical_load_amp": pd.to_numeric(df["electrical_load_amp"], errors="coerce").fillna(0.0),
            "cabin_pressure_ft": pd.to_numeric(df["cabin_pressure_ft"], errors="coerce").fillna(0.0),
            "angle_of_attack_deg": pd.to_numeric(df["angle_of_attack_deg"], errors="coerce").fillna(0.0),
            "flaps_deg": pd.to_numeric(df["flaps_deg"], errors="coerce").fillna(0.0),
            "spoiler_percent": pd.to_numeric(df["spoiler_percent"], errors="coerce").fillna(0.0),
            "vs_fpm": vs_fpm,
            "heading_sin": np.sin(hd_rad),
            "heading_cos": np.cos(hd_rad),
        })
        return feats, df["timestamp"]

# --------------------------
# Calibrator
# --------------------------
class Calibrator:
    def __init__(self):
        self.q50_ = None
        self.q90_ = None
        self.q99_ = None

    def fit(self, scores: np.ndarray):
        self.q50_ = float(np.quantile(scores, 0.50))
        self.q90_ = float(np.quantile(scores, 0.90))
        self.q99_ = float(np.quantile(scores, 0.99))
        return self

    # def predict_proba(self, score: float) -> float:
    #     if score <= self.q50_:
    #         return 0.05 + 0.15 * ((score - (-1.0)) / (self.q50_ - (-1.0) + 1e-6))
    #     elif score <= self.q90_:
    #         return 0.2 + 0.4 * ((score - self.q50_) / (self.q90_ - self.q50_ + 1e-6))
    #     elif score <= self.q99_:
    #         return 0.6 + 0.3 * ((score - self.q90_) / (self.q99_ - self.q90_ + 1e-6))
    #     else:
    #         return float(0.9 + 0.08 * (1.0 - math.exp(-(score - self.q99_))))


    def predict_proba(self, score: float) -> float:
        # maps raw combined standardized score into a 0..1 calibrated anomaly probability
        if self.q50_ is None:
            # fallback linear mapping if not fit
            return float(1.0 / (1.0 + math.exp(-score)))
        if score <= self.q50_:
            return 0.05 + 0.15 * ((score - (-1.0)) / (self.q50_ - (-1.0) + 1e-6))
        elif score <= self.q90_:
            return 0.2 + 0.4 * ((score - self.q50_) / (self.q90_ - self.q50_ + 1e-6))
        elif score <= self.q99_:
            return 0.6 + 0.3 * ((score - self.q90_) / (self.q99_ - self.q90_ + 1e-6))
        else:
            return float(min(0.999, 0.9 + 0.08 * (1.0 - math.exp(-(score - self.q99_)))))
          
# --------------------------
# Rule Engine
# --------------------------
class RuleEngine:
    def __init__(self, thresholds: Dict[str, Dict[str, Tuple[float, float]]]):
        self.thresholds = thresholds

    def check(self, phase: str, row: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        th = self.thresholds.get(phase, DEFAULT_THRESHOLDS["cruise"])
        troubles = []
        alerts = []
        severity = 0.0

        if "is_first_row" in row and row["is_first_row"]:
            return 0.0, ["Start of flight"], []
        if "is_last_row" in row and row["is_last_row"]:
            return 0.0, ["End of flight"], []
        
        def check_range(name: str, value: Optional[float], label: str):
            nonlocal severity
            low, high = th[label]
            if value is None:
                return
            if value < low:
                troubles.append(name)
                # alerts.append(f"{name} low! ({value:.1f} < {low})")
                severity += min(1.0, (low - value) / (abs(low) + 1e-3))
            elif value > high:
                troubles.append(name)
                # alerts.append(f"{name} high! ({value:.1f} > {high})")
                severity += min(1.0, (value - high) / (abs(high) + 1e-3))

        e1_rpm = safe_float(row.get("engine1_rpm"))
        e2_rpm = safe_float(row.get("engine2_rpm"))
        e1_t = safe_float(row.get("engine1_temp_C"))
        e2_t = safe_float(row.get("engine2_temp_C"))


        rpm_avg = np.nanmean([e1_rpm, e2_rpm])
        temp_avg = np.nanmean([e1_t, e2_t])

        # check_range("Engine RPM", rpm_avg, "engine_rpm")
        # check_range("Engine temperature", temp_avg, "engine_temp_C")
        # check_range("Vibration", safe_float(row.get("vibration_mm_s")), "vibration_mm_s")
        # check_range("Fuel flow", safe_float(row.get("fuel_flow_kg_h")), "fuel_flow_kg_h")
        # check_range("Oil pressure", safe_float(row.get("oil_pressure_psi")), "oil_pressure_psi")
        # check_range("Hydraulic pressure", safe_float(row.get("hydraulic_pressure_psi")), "hydraulic_pressure_psi")
        # check_range("Electrical load", safe_float(row.get("electrical_load_amp")), "electrical_load_amp")
        # check_range("Cabin pressure altitude", safe_float(row.get("cabin_pressure_ft")), "cabin_pressure_ft")
        # check_range("Angle of attack", safe_float(row.get("angle_of_attack_deg")), "angle_of_attack_deg")
        # check_range("Flaps", safe_float(row.get("flaps_deg")), "flaps_deg")
        # check_range("Spoilers", safe_float(row.get("spoiler_percent")), "spoiler_percent")

        check_range("Engine 1 RPM", e1_rpm, "engine1_rpm")
        check_range("Engine 2 RPM", e2_rpm, "engine2_rpm")
        check_range("Engine 1 temperature", e1_t, "engine1_temp_C")
        check_range("Engine 2 temperature", e2_t, "engine2_temp_C")
        check_range("Vibration", safe_float(row.get("vibration_mm_s")), "vibration_mm_s")
        check_range("Fuel flow", safe_float(row.get("fuel_flow_kg_h")), "fuel_flow_kg_h")
        check_range("Oil pressure", safe_float(row.get("oil_pressure_psi")), "oil_pressure_psi")
        check_range("Hydraulic pressure", safe_float(row.get("hydraulic_pressure_psi")), "hydraulic_pressure_psi")
        check_range("Electrical load", safe_float(row.get("electrical_load_amp")), "electrical_load_amp")
        check_range("Cabin pressure", safe_float(row.get("cabin_pressure_ft")), "cabin_pressure_ft")
        check_range("Angle of attack", safe_float(row.get("angle_of_attack_deg")), "angle_of_attack_deg")
        check_range("Flaps", safe_float(row.get("flaps_deg")), "flaps_deg")
        check_range("Spoilers", safe_float(row.get("spoiler_percent")), "spoiler_percent")

        if e1_rpm is not None and e2_rpm is not None and abs(e1_rpm - e2_rpm) > 1500:
            troubles.append("Engine RPM imbalance")
            # alerts.append(f"Engine RPM imbalance! Δ={abs(e1_rpm - e2_rpm):.0f}")
            severity += 0.5
        if e1_t is not None and e2_t is not None and abs(e1_t - e2_t) > 80:
            troubles.append("Engine temp imbalance")
            # alerts.append(f"Engine temperature imbalance! Δ={abs(e1_t - e2_t):.0f}°C")
            severity += 0.4

        alt = safe_float(row.get("altitude"))
        cabin = safe_float(row.get("cabin_pressure_ft"))
        if alt is not None and cabin is not None:
            if cabin > 8000:
                troubles.append("Cabin pressure high")
                # alerts.append(f"Cabin altitude high ({cabin:.0f} ft)")
                severity += 0.6
            if cabin > (alt + 3000) and alt < 5000:
                troubles.append("Cabin pressure anomaly")
                # alerts.append("Cabin pressure higher than ambient — possible pressurization issue")
                severity += 0.3

        severity = float(min(3.0, severity * 0.5))  # cut severity in half
        return severity, sorted(set(troubles)), sorted(set(alerts))

# --------------------------
# Accident Risk Model
# --------------------------
class AccidentRiskModel:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.scaler = RobustScaler()
        self.iso = IsolationForest(n_estimators=300, contamination=0.005, random_state=42, n_jobs=-1)

        self.calibrator = Calibrator()
        self.thresholds = DEFAULT_THRESHOLDS
        self.ae_input_dim = None
        self.ae = None

    def _build_autoencoder(self, input_dim: int) -> tf.keras.Model:
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Normalization()(inputs)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        z = layers.Dense(8, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(z)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(input_dim, activation=None)(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, df: pd.DataFrame):

        self.fe.fit(df)
        X = self.fe.transform(df)
        self.ae_input_dim = X.shape[1]

        Xs = self.scaler.fit_transform(X)
        self.iso.fit(Xs)
        dump(self.iso, os.path.join(ARTIFACT_DIR, "iso.pkl"))
        iso_score = -self.iso.score_samples(Xs)

        self.ae = self._build_autoencoder(self.ae_input_dim)
        X_train, X_val = train_test_split(Xs, test_size=0.2, random_state=42)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.ae.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=500, batch_size=256, verbose=0, callbacks=[es])
        recon = self.ae.predict(Xs, verbose=0)
        ae_err = np.mean((Xs - recon) ** 2, axis=1)

        # Rule-based severity
        rule_engine = RuleEngine(self.thresholds)
        severities = []
        for _, row in df.iterrows():
            alt = safe_float(row.get("altitude"))
            phase = "cruise" if (alt is not None and alt >= 10000) else "climb"
            s, _, _ = rule_engine.check(phase, row.to_dict())
            severities.append(s)
        severities = np.array(severities, dtype=np.float32)

        # combined = 0.5 * standardize(iso_score) + 0.35 * standardize(ae_err) + 0.15 * standardize(severities)
        # self.calibrator.fit(combined)
        combined = 0.5 * standardize(iso_score) + \
            0.35 * standardize(ae_err) + \
            0.15 * standardize(severities)
        # Save standardization parameters for inference
        std_params = {
            "iso": {
                "median": float(np.median(iso_score)),
                "mad": float(np.median(np.abs(iso_score - np.median(iso_score))))
            },
            "ae": {
                "median": float(np.median(ae_err)),
                "mad": float(np.median(np.abs(ae_err - np.median(ae_err))))
            },
            "sev": {
                "median": float(np.median(severities)),
                "mad": float(np.median(np.abs(severities - np.median(severities))))
            }
        }
        dump(std_params, os.path.join(ARTIFACT_DIR, "std_params.pkl"))

        self.calibrator.fit(combined)
        dump(self.calibrator, os.path.join(ARTIFACT_DIR, "calibrator.pkl"))

        # Save artifacts
        ensure_artifacts_dir()
        self.ae.save(AE_PATH)
        dump(self.scaler, SCALER_PKL)
        dump(self.fe, MODEL_PKL)  # Contains feature names and encoder
        save_json(self.fe.feature_names_, FEATURES_JSON)
        save_json(self.thresholds, THRESHOLDS_JSON)
        save_json({"ae_input_dim": self.ae_input_dim}, META_JSON)

# denger level = 

    def score_row(self, row: Dict[str, Any], phase_detector: Optional[PhaseDetector] = None) -> Dict[str, Any]:
        phase = "cruise"
        if phase_detector is not None:
            phase, _ = phase_detector.detect(row)

        df = pd.DataFrame([row])
        X = self.fe.transform(df)
        Xs = self.scaler.transform(X)

        iso_score = -self.iso.score_samples(Xs)[0]
        recon = self.ae.predict(Xs, verbose=0)
        ae_err = float(np.mean((Xs - recon) ** 2))

        rule_engine = RuleEngine(self.thresholds)
        severity, troubles, alerts = rule_engine.check(phase, row)

        combined = 0.5 * standardize([iso_score])[0] + \
                0.35 * standardize([ae_err])[0] + \
                0.15 * standardize([severity])[0]

        prob = self.calibrator.predict_proba(combined)
        danger_level = round(prob * 10, 2)

        return {
            "timestamp": row.get("timestamp"),
            "phase": phase,
            "danger_level": danger_level,
            "troubles": troubles,
            "alerts": alerts,
        }

# --------------------------
# CSV / synthetic helpers
# --------------------------
def read_csv_sanitized(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = sanitize_columns(list(df.columns))
    for raw, sani in zip(RAW_COLUMNS, SANITIZED_COLUMNS):
        if sani not in df.columns:
            df[sani] = np.nan
    keep = list(dict.fromkeys(["timestamp"] + SANITIZED_COLUMNS))
    df =  df[[c for c in keep if c in df.columns]]

    df["is_first_row"] = False
    df["is_last_row"] = False
    if not df.empty:
        df.loc[df.index[0], "is_first_row"] = True
        df.loc[df.index[-1], "is_last_row"] = True
    return df

# Synthesize data creation function. 

# def synthesize_data(n: int = 50000, seed: int = 42) -> pd.DataFrame:
#     rng = np.random.default_rng(seed)
#     phases = rng.choice(PHASES, size=n, p=[0.15, 0.25, 0.45, 0.15])
#     base_time = pd.Timestamp.utcnow().floor("s")
#     rows = []
#     alt = 0.0
#     for i in range(n):
#         ph = phases[i]
#         t = base_time + pd.Timedelta(seconds=i)
#         if ph == "ground_takeoff":
#             alt = max(0, alt + rng.normal(20, 10))
#             flaps = rng.uniform(5, 15)
#             spoiler = rng.uniform(0, 2)
#         elif ph == "climb":
#             alt = max(1500, alt + rng.normal(80, 30))
#             flaps = rng.uniform(0, 5)
#             spoiler = rng.uniform(0, 2)
#         elif ph == "cruise":
#             alt = max(10000, alt + rng.normal(0, 5))
#             flaps = rng.uniform(0, 1)
#             spoiler = rng.uniform(0, 1)
#         else:
#             alt = max(0, alt + rng.normal(-100, 40))
#             flaps = rng.uniform(5, 25)
#             spoiler = rng.uniform(0, 40)

#         rpm1 = rng.normal(8000, 1000)
#         rpm2 = rpm1 + rng.normal(0, 300)
#         temp1 = rng.normal(700, 80)
#         temp2 = temp1 + rng.normal(0, 30)
#         vib = abs(rng.normal(1.5, 0.7))
#         fuel = abs(rng.normal(2000, 500))
#         oil = rng.normal(60, 10)
#         hyd = rng.normal(3000, 200)
#         elec = abs(rng.normal(200, 50))
#         cabin = alt - rng.normal(0, 200)
#         aoa = rng.normal(5, 2)
#         hd = rng.uniform(0, 360)

#         rows.append([t, alt, rpm1, rpm2, temp1, temp2, vib, fuel, oil, hd, hyd, elec, cabin, aoa, flaps, spoiler])
#     df = pd.DataFrame(rows, columns=RAW_COLUMNS)
#     df["is_first_row"] = False
#     df["is_last_row"] = False
#     if not df.empty:
#        df.loc[df.index[0], "is_first_row"] = True
#        df.loc[df.index[-1], "is_last_row"] = True
# return df

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()

    ARTIFACT_DIR = args.out_dir
    ensure_artifacts_dir()

    if args.train:
        if args.csv:
            df = read_csv_sanitized(args.csv)
        # elif args.synthetic:
            # df = synthesize_data(n=5000)
        else:
            print("Provide --csv or --synthetic for training")
            return

        model = AccidentRiskModel()
        model.fit(df)
        print(f"Training complete. Artifacts saved in {ARTIFACT_DIR}")

if __name__ == "__main__":
    main()
