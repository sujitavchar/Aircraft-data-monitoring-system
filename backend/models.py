from  pydantic import BaseModel
from typing import List, Any

# Request structure for  flight data row
class DataRow(BaseModel):
    timestamp: str
    altitude : float
    engine1_rpm: float
    engine2_rpm: float
    engine1_temp_C: float
    engine2_temp_C: float
    vibration_mm_s: float
    fuel_flow_kg_h: float
    oil_pressure_psi: float
    heading_deg: float
    hydraulic_pressure_psi: float
    electrical_load_amp: float 
    cabin_pressure_ft: float
    angle_of_attack_deg: float
    flaps_deg: float
    spoiler_percent: float


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