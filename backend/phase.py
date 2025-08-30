def get_phase(row):
    alt = row["altitude_ft"]
    vs = row["vertical_speed_fpm"]
    flaps = row["flaps_deg"]
    spoilers = row["spoiler_percent"]
    spd = row["airspeed_kts"]
    pitch = row["pitch_deg"]

    # Takeoff - Low altitude, increasing vertical speed, speed above stall threshold, flaps partially deployed
    if alt<5000 and spd > 130 and flaps >= 5 and vs > 500 and pitch > 5:
        return "takeoff"

    #Climb - Positive vertical speed, moderate pitch, speed building, flaps retracted or minimal
    if alt >=5000 and vs > 500 and flaps <= 5 and spd > 200 and pitch > 0:
        return "climb"

    # Cruise - High altitude, stable speed, near-zero vertical speed, flaps and spoilers retracted
    if alt >= 28000 and -300 <= vs <= 300 and flaps == 0 and spoilers == 0 and 250 <= spd <= 550:
        return "cruise"

    # Descent - Negative vertical speed, altitude still above landing threshold, flaps/spoilers mostly retracted
    if alt > 3000 and vs < -500 and flaps <= 5 and spd > 180:
        return "descent"

    # Landing - Low altitude, reduced speed, high flap setting, descent or approach
    if alt <= 3000 and flaps >= 20 and spd < 180 and vs <= 0:
        return "landing"

    # For other values
    return "unknown"
