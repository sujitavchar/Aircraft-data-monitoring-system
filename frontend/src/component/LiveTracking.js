import React, { useState, useEffect, useRef } from "react";
import Papa from "papaparse";
import LineGraph from "./LineGraph";
import "typeface-roboto-mono";
import "../style/Livetracking.css";

import livetracking from "../data/safe_flight_.csv";
import AreaGraph from "./AreaGraph";

function LiveTrackingPage() {
  const [csvData, setcsvData] = useState([]);
  const [altitude, setaltitude] = useState([]);
  const [engine1rpm, setengine1rpm] = useState([]);
  const [engine2rpm, setengine2rpm] = useState([]);
  const [temp, settemp] = useState([]);
  const [temp1, settemp1] = useState([]);
  const [vibration, setvibration] = useState([]);
  const [fuelFlow, setfuelFlow] = useState([]);
  const [oilPressure, setoilPressure] = useState([]);
  const [hydraulicPressure, sethydraulicPressure] = useState([]);
  const [flightDate, setFlightDate] = useState("");

  const [parameters, setparameters] = useState({
    Danger: 0,
    Probability: 0,
    Trouble_Parameters: 0,
    Phase: "N/A",
  });

  const [flightTime, setflightTime] = useState({
    hours: 0,
    minutes: 0,
    seconds: 0,
  });

  const currentIndex = useRef(0);
  const MAX_POINTS = 60;
  const wsRef = useRef(null);

  function stringToJson(data2) {
    try {
      if (
        (data2.startsWith("'") && data2.endsWith("'")) ||
        (data2.startsWith('"') && data2.endsWith('"'))
      ) {
        data2 = data2.slice(1, -1);
      }
      return JSON.parse(data2);
    } catch (error) {
      console.error("Invalid JSON string:", error.message);
      return null;
    }
  }


  const dangerValue = String(parameters.Danger).includes("/")
    ? Number(String(parameters.Danger).split("/")[0])
    : Number(parameters.Danger) || 0;

  const dangerFillPercent = Math.min(
    Math.max((dangerValue / 10) * 100, 0),
    100
  );

  const getDangerClass = (danger) => {
    const val = String(danger).includes("/")
      ? Number(String(danger).split("/")[0])
      : Number(danger);

    if (val < 4) return "safe";        
    if (val < 7) return "moderate";    
    if (val < 8) return "warning";     
    return "critical";                
  };

  const getProbabilityClass = (probability) => {
  const val = Number(probability) * 100 || 0;

  if (val < 40) return "safe";       
  if (val < 70) return "moderate";    
  if (val < 80) return "warning";     
  return "critical";                 
};

  const troubleList = Array.isArray(parameters.Trouble_Parameters)
    ? parameters.Trouble_Parameters
    : String(parameters.Trouble_Parameters)
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);

  const probabilityValue = Number(parameters.Probability) || 0;
  const probabilityDisplay = (probabilityValue * 100).toFixed(1);
  const probabilityFillPercent = Math.min(probabilityValue * 100, 100);

  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/live-monitoring");
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const parsed_data = stringToJson(data.model_response);

      console.log("Received WebSocket message:", parsed_data);

      setparameters({
        Danger: parsed_data["Danger Level"] || "NULL",
        Probability: parsed_data["Crash Probability"] || "NULL",
        Trouble_Parameters: parsed_data["Trouble Params"] || "NULL",
        Phase: parsed_data["Phase"] || "NULL",
      });
    };

    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    fetch(livetracking)
      .then((res) => res.text())
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            setcsvData(results.data);
            currentIndex.current = 0;

            if (results.data.length > 0 && results.data[0].timestamp) {
              const [datePart, timePart] =
                results.data[0].timestamp.split(" ");
              setFlightDate(datePart);

              const [h, m, s] = timePart.split(":").map(Number);
              setflightTime({ hours: h, minutes: m, seconds: s });
            }
          },
        });
      });
  }, []);

  useEffect(() => {
    let rowss = [];
    for (let i = 0; i < MAX_POINTS; i++) {
      rowss.push({
        time: `T${i + 1}`,
        altitude_ft: null,
      });
    }
    setaltitude(rowss);
  }, []);

  useEffect(() => {
    if (csvData.length === 0) return;

    const interval = setInterval(() => {
      if (currentIndex.current >= csvData.length) {
        clearInterval(interval);
        return;
      }

      const row = csvData[currentIndex.current];
      const timeLabel = row.timestamp || `T${currentIndex.current + 1}`;

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(row));
      }

      setflightTime((prev) => {
        let { hours, minutes, seconds } = prev;
        seconds += 1;
        if (seconds >= 60) {
          seconds = 0;
          minutes += 1;
        }
        if (minutes >= 60) {
          minutes = 0;
          hours += 1;
        }
        return { hours, minutes, seconds };
      });

      setaltitude((prev) => {
        const updated = [...prev];
        if (currentIndex.current < MAX_POINTS) {
          updated[currentIndex.current] = {
            time: timeLabel,
            altitude: Number(row.altitude),
          };
        } else {
          updated.shift();
          updated.push({
            time: timeLabel,
            altitude: Number(row.altitude),
          });
        }
        return updated;
      });

      setengine1rpm((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, engine1_rpm: Number(row.engine1_rpm) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      setengine2rpm((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, engine2_rpm: Number(row.engine2_rpm) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      settemp((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, engine1_temp_C: Number(row.engine1_temp_C) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      settemp1((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, engine2_temp_C: Number(row.engine2_temp_C) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      setvibration((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, vibration_mm_s: Number(row.vibration_mm_s) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      setfuelFlow((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, fuel_flow_kg_h: Number(row.fuel_flow_kg_h) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      setoilPressure((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, oil_pressure_psi: Number(row.oil_pressure_psi) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      sethydraulicPressure((prev) => {
        const updated = [
          ...prev,
          {
            time: timeLabel,
            hydraulic_pressure_psi: Number(row.hydraulic_pressure_psi),
          },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      currentIndex.current += 1;
    }, 1000);

    return () => clearInterval(interval);
  }, [csvData]);

  const formatTime = (h, m, s) =>
    `${String(h).padStart(2, "0")}:${String(m).padStart(
      2,
      "0"
    )}:${String(s).padStart(2, "0")}`;

  return (
    <div className="live-tracking-page">
      <div className="graphs">
        <div className="time-display">
          {flightDate}{" "}
          {formatTime(
            flightTime.hours,
            flightTime.minutes,
            flightTime.seconds
          )}
        </div>

        <AreaGraph
          data={altitude}
          xKey="time"
          yKey="altitude"
          lineColor="#FF0000"
          title="Altitude vs Time"
        />
        <LineGraph
          data={engine1rpm}
          xKey="time"
          yKey="engine1_rpm"
          lineColor="#0000FF"
          title="Engine 1 RPM vs Time"
        />
        <LineGraph
          data={engine2rpm}
          xKey="time"
          yKey="engine2_rpm"
          lineColor="#008000"
          title="Engine 2 RPM vs Time"
        />
        <LineGraph
          data={temp}
          xKey="time"
          yKey="engine1_temp_C"
          lineColor="#FF8C00"
          title="Temperature (°C) vs Time"
        />
        <LineGraph
          data={temp1}
          xKey="time"
          yKey="engine2_temp_C"
          lineColor="#FF8C00"
          title="Temperature (°C) vs Time"
        />
        <LineGraph
          data={vibration}
          xKey="time"
          yKey="vibration_mm_s"
          lineColor="#800080"
          title="Vibration (mm/s) vs Time"
        />
        <LineGraph
          data={fuelFlow}
          xKey="time"
          yKey="fuel_flow_kg_h"
          lineColor="#FF1493"
          title="Fuel Flow (kg/h) vs Time"
        />
        <LineGraph
          data={oilPressure}
          xKey="time"
          yKey="oil_pressure_psi"
          lineColor="#00CED1"
          title="Oil Pressure (psi) vs Time"
        />
        <LineGraph
          data={hydraulicPressure}
          xKey="time"
          yKey="hydraulic_pressure_psi"
          lineColor="#FFD700"
          title="Hydraulic Pressure (psi) vs Time"
        />
      </div>


      <div className="parameters">
        <h2 className="flight-status-title">FLIGHT STATUS</h2>

       
        <div className="param-box">
          <div className="param-head">
            <span className="param-label">Danger Level</span>
          </div>

          <div className="danger-card">
            <div className="danger-top">
              <div className="danger-number">{parameters.Danger}</div>
              <div
                className={`danger-pill ${getDangerClass(parameters.Danger)}`}
              >
                {getDangerClass(parameters.Danger).toUpperCase()}
              </div>
            </div>

            <div className="danger-subtext">Danger Scale (0 - 10)</div>

            <div className="danger-bar-dark">
              <div
                className={`danger-fill-dark ${getDangerClass(parameters.Danger)}`}
                style={{ width: `${dangerFillPercent}%` }}
              ></div>
            </div>
          </div>
        </div>

   
        <div className="param-box">
          <div className="param-head">
            <span className="param-label">Probability</span>
          </div>

          <div className="probability-card">
            <div className="prob-value">{probabilityDisplay}%</div>
            <div className="prob-text">Crash Probability</div>

            <div className="prob-bar">
              <div
                className={`prob-fill ${getProbabilityClass(parameters.Probability)}`}
                style={{ width: `${probabilityFillPercent}%` }}
              ></div>
            </div>
          </div>
        </div>

      
        <div className="param-box">
          <div className="param-head">
            <span className="param-label">Phase</span>
          </div>

          <div className="phase-box">{parameters.Phase}</div>
        </div>


        <div className="param-box">
          <div className="param-head">
            <span className="param-label">Trouble Parameters</span>
          </div>

          <div className="trouble-list">
            {troubleList.length > 0 && parameters.Trouble_Parameters !== "NULL" ? (
              troubleList.map((item, index) => (
                <div key={index} className="trouble-item">
                  {item}
                </div>
              ))
            ) : (
              <div className="no-trouble">No Trouble Issues</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default LiveTrackingPage;