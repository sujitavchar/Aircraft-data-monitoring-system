import React, { useState, useEffect, useRef } from "react";
import Papa from "papaparse";
import LineGraph from "./LineGraph";
import 'typeface-roboto-mono';
import '../style/Livetracking.css';

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
  // const [engineTorque, setengineTorque] = useState([]);
  const [flightDate, setFlightDate] = useState(""); 

  const [parameters, setparameters] = useState({
    timestamp: "",
    danger: 0,
    probability: 0,
    aluminum: 0,
    report: "N/A"
  });

  const [flightTime, setflightTime] = useState({ hours: 0, minutes: 0, seconds: 0 });
  const currentIndex = useRef(0);
  const MAX_POINTS = 60;

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
          const [datePart, timePart] = results.data[0].timestamp.split(" ");
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
          altitude_ft: null
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

      // Update timer
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
          altitude: Number(row.altitude)
        };
        } else {
          updated.shift(); 
          updated.push({
            time: timeLabel,
            altitude: Number(row.altitude)
          });
        }
        return updated;
      });

      setengine1rpm((prev) => {
        const updated = [...prev, { time: timeLabel, engine1_rpm: Number(row.engine1_rpm) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });


      setengine2rpm((prev) => {
        const updated = [...prev, { time: timeLabel, engine2_rpm: Number(row.engine2_rpm) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });


      settemp((prev) => {
        const updated = [...prev, { time: timeLabel, engine1_temp_C: Number(row.engine1_temp_C) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      settemp1((prev) => {
        const updated = [...prev, { time: timeLabel, engine2_temp_C: Number(row.engine2_temp_C) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });


      setvibration((prev) => {
        const updated = [...prev, { time: timeLabel, vibration_mm_s: Number(row.vibration_mm_s) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });


      setfuelFlow((prev) => {
        const updated = [...prev, { time: timeLabel, fuel_flow_kg_h: Number(row.fuel_flow_kg_h) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });


      setoilPressure((prev) => {
        const updated = [...prev, { time: timeLabel, oil_pressure_psi: Number(row.oil_pressure_psi) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });


      sethydraulicPressure((prev) => {
        const updated = [...prev, { time: timeLabel, hydraulic_pressure_psi: Number(row.hydraulic_pressure_psi) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });


      // setengineTorque((prev) => {
      //   const updated = [...prev, { time: timeLabel, engine_torque_nm: Number(row.engine_torque_nm) }];
      //   if (updated.length > MAX_POINTS) updated.shift();
      //   return updated;
      // });

      setparameters({
        timestamp: row.timestamp || timeLabel,
        danger: Number(row.danger_level) || 0,
        probability: Number(row.probability) || 0,
        aluminum: Number(row.aluminum) || 0,
        report: row.report || "N/A"
      });

      currentIndex.current += 1;
    }, 1000);

    return () => clearInterval(interval);
  }, [csvData]);

  const formatTime = (h, m, s) =>
    `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;

  return (
    <div className="live-tracking-page">
      
      <div className="graphs">
        <div className="time-display">
          {flightDate} {formatTime(flightTime.hours, flightTime.minutes, flightTime.seconds)}
        </div>
        <AreaGraph data={altitude} xKey="time" yKey="altitude" lineColor="#FF0000" title="Altitude vs Time" />
        <LineGraph data={engine1rpm} xKey="time" yKey="engine1_rpm" lineColor="#0000FF" title="Engine 1 RPM vs Time" />
        <LineGraph data={engine2rpm} xKey="time" yKey="engine2_rpm" lineColor="#008000" title="Engine 2 RPM vs Time" />
        <LineGraph data={temp} xKey="time" yKey="engine1_temp_C" lineColor="#FF8C00" title="Temperature (°C) vs Time" />
        <LineGraph data={temp1} xKey="time" yKey="engine2_temp_C" lineColor="#FF8C00" title="Temperature (°C) vs Time" />
        <LineGraph data={vibration} xKey="time" yKey="vibration_mm_s" lineColor="#800080" title="Vibration (mm/s) vs Time" />
        <LineGraph data={fuelFlow} xKey="time" yKey="fuel_flow_kg_h" lineColor="#FF1493" title="Fuel Flow (kg/h) vs Time" />
        <LineGraph data={oilPressure} xKey="time" yKey="oil_pressure_psi" lineColor="#00CED1" title="Oil Pressure (psi) vs Time" />
        <LineGraph data={hydraulicPressure} xKey="time" yKey="hydraulic_pressure_psi" lineColor="#FFD700" title="Hydraulic Pressure (psi) vs Time" />
        {/* <LineGraph data={engineTorque} xKey="time" yKey="engine_torque_nm" lineColor="#32CD32" title="Engine Torque (Nm) vs Time" /> */}
      </div>

      {/* LEFT SIDE (fixed parameters) */}
      <div className="parameters">
        <h2>Parameters</h2>
        <div><strong>Timestamp:</strong> {parameters.timestamp}</div>
        <div><strong>Danger Level:</strong> {parameters.danger}</div>
        <div><strong>Probability:</strong> {parameters.probability}%</div>
        <div><strong>Aluminum:</strong> {parameters.aluminum}</div>
        <div><strong>Report:</strong> {parameters.report}</div>
      </div>
    </div>
  );
}

export default LiveTrackingPage;
