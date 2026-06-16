import React, { useState, useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";
import Papa from "papaparse";
import LineGraph from "./LineGraph";
import AreaGraph from "./AreaGraph";

import 'typeface-roboto-mono';
import '../style/GraphsPages.css';

function GraphsPages() {
  const location = useLocation();
  const file = location.state?.fileData;
  const [csvData, setCsvData] = useState([]);
  const [altitude, setAltitude] = useState([]);
  const [airspeed, setAirspeed] = useState([]);
  const [pitchRoll, setPitchRoll] = useState([]);
  const [verticalSpeed, setVerticalSpeed] = useState([]);
  const [engineTemp, setEngineTemp] = useState([]);
  const [spoilerFlaps, setSpoilerFlaps] = useState([]);
  const [flightTime, setFlightTime] = useState({ hours: 0, minutes: 0, seconds: 0 });
  const currentIndex = useRef(0);
  const MAX_POINTS = 60;

  const [flightDate, setFlightDate] = useState(""); 
  useEffect(() => {
    if (!file) return;

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        setCsvData(results.data);
        currentIndex.current = 0;

        if (results.data.length > 0 && results.data[0].timestamp) {
          const [datePart, timePart] = results.data[0].timestamp.split("T");
          setFlightDate(datePart); 
          const cleanTime = timePart.replace("Z", ""); 
          const [h, m, s] = cleanTime.split(":").map(Number);
          setFlightTime({ hours: h, minutes: m, seconds: s });
        }
      },
    });
  }, [file]);


  //const MAX_POINTS = 100;

    useEffect(() => {
      let rowss = [];
      for (let i = 0; i < MAX_POINTS; i++) {
        rowss.push({
          time: `T${i + 1}`,
          altitude_ft: null
        });
      }
      setAltitude(rowss);
    }, []);


  // Update graphs and timer every second
  useEffect(() => {
    if (csvData.length === 0) return;

    const interval = setInterval(() => {
      if (currentIndex.current >= csvData.length) {
        clearInterval(interval);
        return;
      }

      const row = csvData[currentIndex.current];
      const timeLabel = `T${currentIndex.current + 1}`;

      // Update timer
      setFlightTime((prev) => {
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

      // Update graphs
      setAltitude((prev) => {
        const updated = [...prev];
        if (currentIndex.current < MAX_POINTS) {
          updated[currentIndex.current] = {
          time: `T${currentIndex.current + 1}`,
          altitude_ft: Number(row.altitude_ft)
        };
        } else {
          updated.shift(); 
          updated.push({
            time: `T${currentIndex.current + 1}`,
            altitude_ft: Number(row.altitude_ft)
          });
        }
        return updated;
      });

      setAirspeed((prev) => {
        const updated = [...prev, { time: timeLabel, airspeed_kts: Number(row.airspeed_kts) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      setPitchRoll((prev) => {
        const updated = [...prev,{ time: timeLabel, pitch_deg: Number(row.pitch_deg), roll_deg: Number(row.roll_deg) },];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      setVerticalSpeed((prev) => {
        const updated = [...prev, { time: timeLabel, vertical_speed_fpm: Number(row.vertical_speed_fpm) }];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      setEngineTemp((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, engine1_temp_C: Number(row.engine1_temp_C), engine2_temp_C: Number(row.engine2_temp_C) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      setSpoilerFlaps((prev) => {
        const updated = [
          ...prev,
          { time: timeLabel, spoiler_percent: Number(row.spoiler_percent), flaps_deg: Number(row.flaps_deg) },
        ];
        if (updated.length > MAX_POINTS) updated.shift();
        return updated;
      });

      currentIndex.current += 1;
    }, 1000);

    return () => clearInterval(interval);
  }, [csvData]);

  const formatTime = (h, m, s) =>
  `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;

  return (
    <div className="graphs-page">
      
      <div
        style={{position: 'fixed',top: 20, left: '50%', transform: 'translateX(-50%)', padding: '5px 15px',
          backgroundColor: 'rgba(0, 0, 0, 0.5)', color: '#ffffff', fontFamily: 'monospace', fontSize: '22px', borderRadius: '5px',
          zIndex: 9999, textAlign: 'center', }}>
          {flightDate} {formatTime(flightTime.hours, flightTime.minutes, flightTime.seconds)}
      </div>

      <div className="graphs-container">
        <div className="graph-wrapper">
          <AreaGraph data={altitude} xKey="time" yKey="altitude_ft" lineColor="#FF0000" title="Altitude (ft) vs Time" />
        </div>
        <div className="graph-wrapper">
          <LineGraph data={airspeed} xKey="time" yKey="airspeed_kts" lineColor="#0000FF" title="Airspeed (kts) vs Time" />
        </div>
        <div className="graph-wrapper">
          <LineGraph data={engineTemp} xKey="time" yKey="engine1_temp_C" lineColor="#008080" title="Engine1 Temp (°C) vs Time" />
        </div>
        <div className="graph-wrapper">
          <LineGraph data={engineTemp} xKey="time" yKey="engine2_temp_C" lineColor="#808000" title="Engine2 Temp (°C) vs Time" />
        </div>
        <div className="graph-wrapper">
          <LineGraph data={verticalSpeed} xKey="time" yKey="vertical_speed_fpm" lineColor="#0509ffff" title="Vertical Speed (fpm) vs Time" />
        </div>
        <div className="graph-wrapper">
          <LineGraph data={pitchRoll} xKey="time" yKey="pitch_deg" lineColor="#00FF00" title="Pitch (deg) vs Time" />
        </div>
        <div className="graph-wrapper">
          <LineGraph data={pitchRoll} xKey="time" yKey="roll_deg" lineColor="#FFA500" title="Roll (deg) vs Time" />
        </div>
        <div className="graph-wrapper">
          <LineGraph data={spoilerFlaps} xKey="time" yKey="flaps_deg" lineColor="#FF1493" title="Flaps (deg) vs Time" />
        </div>
      </div>
    </div>
  );
}

export default GraphsPages;
