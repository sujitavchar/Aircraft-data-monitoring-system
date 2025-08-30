from  pydantic import BaseModel
from typing import List, Any

# Request structure for  flight data row
class DataRow(BaseModel):
    timestamp: str
    altitude_ft : float
    airspeed_kts: float
    pitch_deg: float
    roll_deg: float
    vertical_speed_fpm: float
    engine1_temp_C: float
    engine2_temp_C : float
    heading_deg: float
    flaps_deg: float
    spoiler_percent: float
    phase: str

# request structure for anomaly 
class Anomaly(BaseModel):
    timestamp: str
    parameter: str
    value: Any
    rule: str
    phase: str

# Response structure for sending report
class AnalysisReportResponse(BaseModel):
    file: str
    total_rows: int
    total_anomalies: int
    report_text: str