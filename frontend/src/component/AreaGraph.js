import React from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import '../style/GraphsPages.css';

function AreaGraph({ data, xKey, yKey }) {
  return (
    <div
      style={{
        border: "0.5px solid #555", 
        padding: "10px",
        borderRadius: "5px",
        backgroundColor: "#000"
      }}
    >
      <h3 className="graph-title">Altitude (ft) vs Time</h3>

      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorAltitude" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#FF0000" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#FF0000" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#333" strokeDasharray="3 3" />
          <XAxis
            dataKey={xKey}
            tick={false}     // hide all ticks
            aaxisLine={{ stroke: '#ccc' }}
          />


          <YAxis tick={{ fill: "#ccc" }} stroke="#c62020ff"/>
          <Tooltip contentStyle={{ backgroundColor: "#000", border: "0.5px solid #555", color: "#fff" }}/>
          <Area type="monotone" dataKey={yKey} stroke="#FF0000" fill="url(#colorAltitude)" isAnimationActive={false} connectNulls={true}/>
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export default AreaGraph;
