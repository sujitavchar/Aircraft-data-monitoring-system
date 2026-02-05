from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
import os, uuid, shutil
import tensorflow as tf

from pathlib import Path
from backend.parser import parse_csv
from backend.report import generate_report
from starlette.concurrency import run_in_threadpool
import numpy as np
import pandas as pd
import json
import sys

from model.execute import predict
from fastapi.middleware.cors import CORSMiddleware


ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

app = FastAPI(title = "Aircraft-data-monitoring-system", version="1.0")


origins = [
    "http://localhost:3000",   # React frontend
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Allowed domains
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],          # Allow all headers
)


#Pathlib for safer paths
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



@app.get("/")
def read_root():
    return {"message": "Welcome to Root"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code = 400, detail = "Only csv files  are allowed")
    
    safe_name = Path(file.filename).name
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"

    file_path = UPLOAD_DIR / unique_name  # assigns an unique name to file to avoid clashes


    #save file asynchronously - offloading IO buffer
    try:
        with open(file_path, "wb") as buffer:
            await run_in_threadpool(shutil.copyfileobj, file.file, buffer)

    finally:
        file.file.close()

    # parse and indentify anomalies
    try:
        rows, anomalies = await run_in_threadpool(parse_csv, file_path)

    except Exception as e:
        raise HTTPException(status_code= 500, detail= f"File cannot be parsed : {e}")
    
    try:
        report_text = await run_in_threadpool(generate_report, anomalies)

    except Exception as e:
        raise HTTPException(status_code= 500, detail= f"Error is generating report : {e}")
    
    #clears server memory after processing the file
    file_path.unlink(missing_ok=True)

    return {
        "file": safe_name,
        "total_rows": len(rows),
        "total_anomalies": len(anomalies),
        "report_text": report_text
    }


@app.websocket("/ws/live-monitoring")
async def ws_live_monitoring(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_json()

                # Validate required keys
                required_keys = [
                    "timestamp", "altitude", "engine1_rpm", "engine2_rpm",
                    "engine1_temp_C", "engine2_temp_C", "vibration_mm_s",
                    "fuel_flow_kg_h", "oil_pressure_psi", "heading_deg",
                    "hydraulic_pressure_psi", "electrical_load_amp",
                    "cabin_pressure_ft", "angle_of_attack_deg",
                    "flaps_deg", "spoiler_percent"
                ]
                for key in required_keys:
                    if key not in data:
                        await websocket.send_json({
                            "success": False,
                            "error": f"Missing field: {key}"
                        })
                        continue

                # # Convert JSON to numpy array (excluding timestamp if not numeric)
                # input_array = [
                #     data["timestamp"],  # string
                #     float(data["altitude"]),
                #     float(data["engine1_rpm"]),
                #     float(data["engine2_rpm"]),
                #     float(data["engine1_temp_C"]),
                #     float(data["engine2_temp_C"]),
                #     float(data["vibration_mm_s"]),
                #     float(data["fuel_flow_kg_h"]),
                #     float(data["oil_pressure_psi"]),
                #     float(data["heading_deg"]),
                #     float(data["hydraulic_pressure_psi"]),
                #     float(data["electrical_load_amp"]),
                #     float(data["cabin_pressure_ft"]),
                #     float(data["angle_of_attack_deg"]),
                #     float(data["flaps_deg"]),
                #     float(data["spoiler_percent"])
                # ]

                # Run prediction
                try:
                    prediction = predict(data)
                    
                    await websocket.send_json({
                        "success": True,
                        "timestamp": data["timestamp"],  
                        "model_response": prediction
                    })
                except Exception as e:
                    await websocket.send_json({
                        "success": False,
                        "error": f"Model prediction error: {str(e)}"
                    })

            except ValueError as e:  # bad JSON
                await websocket.send_json({
                    "success": False,
                    "error": f"Invalid JSON: {str(e)}"
                })
            except Exception as e:
                await websocket.send_json({
                    "success": False,
                    "error": f"Unexpected error: {str(e)}"
                })

    except WebSocketDisconnect:
        print("Client disconnected")