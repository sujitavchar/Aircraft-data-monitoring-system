import csv
from  rules_engine import detect_anomalies
from phase import get_phase


def parse_csv(file_path: str):
    anomalies = []
    rows = []

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        required_cols = {
            "timestamp", "altitude_ft", "airspeed_kts", "pitch_deg", "roll_deg","vertical_speed_fpm", "engine1_temp_C", "engine2_temp_C",
            "heading_deg", "flaps_deg", "spoiler_percent"
        }

        missing_cols = required_cols - set(reader.fieldnames or [])

        if missing_cols:
            raise ValueError(f"Missing columms in data: {missing_cols}")
        
        for row in reader:
            try:
                parsed_row = {
                    "timestamp": row["timestamp"],
                    "altitude_ft": float(row["altitude_ft"]),
                    "airspeed_kts": float(row["airspeed_kts"]),
                    "pitch_deg": float(row["pitch_deg"]),
                    "roll_deg": float(row["roll_deg"]),
                    "vertical_speed_fpm": float(row["vertical_speed_fpm"]),
                    "engine1_temp_C": float(row["engine1_temp_C"]),
                    "engine2_temp_C": float(row["engine2_temp_C"]),
                    "heading_deg": float(row["heading_deg"]),
                    "flaps_deg": float(row["flaps_deg"]),
                    "spoiler_percent": float(row["spoiler_percent"]),
                }
            except (ValueError, KeyError) as e:
                continue  # if data is bad, continue instead of crashing
            
            # calculate flight phase
            phase = get_phase(parsed_row)
            parsed_row["phase"] = phase

            # detect anomalies
            detected_anomalies = detect_anomalies(parsed_row, phase)
            if detected_anomalies:
                anomalies.extend(detected_anomalies)

            rows.append(parsed_row)

    return rows, anomalies





