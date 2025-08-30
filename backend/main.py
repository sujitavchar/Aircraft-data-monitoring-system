from fastapi import FastAPI, UploadFile, File, HTTPException
import os, uuid, shutil
from pathlib import Path
from parser import parse_csv
from report import generate_report
from starlette.concurrency import run_in_threadpool

app = FastAPI(title = "Aircraft-data-monitoring-system", version="1.0")

#Path;ib for safer paths
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "Welcome to Root"}

@app.put("/upload")
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