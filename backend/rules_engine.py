

def detect_anomalies(row, phase):
    anomalies = []

    # Altitude
   
    if row["altitude_ft"] < 0:
        anomalies.append(make_anomaly(row, "altitude_ft", "Negative altitude", phase))
    if phase == "cruise" and row["altitude_ft"] < 18000:
        anomalies.append(make_anomaly(row, "altitude_ft", "Too low for cruise", phase))
    if phase == "takeoff" and row["altitude_ft"] > 20000:
        anomalies.append(make_anomaly(row, "altitude_ft", "Abnormally high during takeoff", phase))

  
    # Airspeed
    
    if phase == "takeoff" and row["airspeed_kts"] < 130:
        anomalies.append(make_anomaly(row, "airspeed_kts", "Too low for takeoff", phase))
    if phase == "cruise" and not (250 <= row["airspeed_kts"] <= 550):
        anomalies.append(make_anomaly(row, "airspeed_kts", "Unsafe cruise speed", phase))
    if phase == "landing" and row["altitude_ft"] < 3000 and row["airspeed_kts"] > 180:
        anomalies.append(make_anomaly(row, "airspeed_kts", "Too fast for landing", phase))
    if row["airspeed_kts"] <= 0:
        anomalies.append(make_anomaly(row, "airspeed_kts", "Invalid airspeed (stalled)", phase))


    # Pitch

    if not (-15 <= row["pitch_deg"] <= 20):
        if not (phase == "takeoff" and row["pitch_deg"] <= 25) and not (phase == "landing" and row["pitch_deg"] >= -10):
            anomalies.append(make_anomaly(row, "pitch_deg", "Unsafe pitch angle", phase))

 
    # Roll
    
    if abs(row["roll_deg"]) > 45:
        anomalies.append(make_anomaly(row, "roll_deg", "Excessive bank angle", phase))
    if phase == "takeoff" and abs(row["roll_deg"]) > 10:
        anomalies.append(make_anomaly(row, "roll_deg", "Banking during takeoff", phase))

    
    # Vertical Speed

    if row["vertical_speed_fpm"] < -3000:
        anomalies.append(make_anomaly(row, "vertical_speed_fpm", "Uncontrolled descent", phase))
    if row["vertical_speed_fpm"] > 4000:
        anomalies.append(make_anomaly(row, "vertical_speed_fpm", "Abnormal climb", phase))
    if phase == "landing" and row["vertical_speed_fpm"] > -500:
        anomalies.append(make_anomaly(row, "vertical_speed_fpm", "Too shallow descent for landing", phase))

 
    # Engine Temp
  
    if row["engine1_temp_C"] > 720:
        anomalies.append(make_anomaly(row, "engine1_temp_C", "Engine1 overheating", phase))
    if row["engine2_temp_C"] > 720:
        anomalies.append(make_anomaly(row, "engine2_temp_C", "Engine2 overheating", phase))
    if row["engine1_temp_C"] < 300 and row["altitude_ft"] > 1000:
        anomalies.append(make_anomaly(row, "engine1_temp_C", "Engine1 flameout", phase))
    if abs(row["engine1_temp_C"] - row["engine2_temp_C"]) > 100:
        anomalies.append(make_anomaly(row, "engine1_temp_C", "Engine temperature imbalance", phase))

    # Heading
   
    if not (0 <= row["heading_deg"] <= 360):
        anomalies.append(make_anomaly(row, "heading_deg", "Invalid heading", phase))

  
    # Flaps
    
    if phase == "cruise" and row["flaps_deg"] > 0:
        anomalies.append(make_anomaly(row, "flaps_deg", "Flaps deployed in cruise", phase))
    if phase == "takeoff" and row["flaps_deg"] == 0:
        anomalies.append(make_anomaly(row, "flaps_deg", "Flaps not deployed during takeoff", phase))
    if phase == "landing" and not (20 <= row["flaps_deg"] <= 40):
        anomalies.append(make_anomaly(row, "flaps_deg", "Improper flap setting for landing", phase))

    # Spoilers
  
    if phase in ["takeoff", "cruise", "climb"] and row["spoiler_percent"] > 10:
        anomalies.append(make_anomaly(row, "spoiler_percent", "Spoilers deployed mid-flight", phase))
    if phase == "landing" and not (0 <= row["spoiler_percent"] <= 60):
        anomalies.append(make_anomaly(row, "spoiler_percent", "Improper spoiler usage", phase))

    return anomalies


def make_anomaly(row, param, rule, phase):
    return {
        "timestamp": row["timestamp"],
        "parameter": param,
        "value": row[param],
        "rule": rule,
        "phase": phase
    }
