import React from "react";
import {LineChart,Line,XAxis,YAxis,Tooltip,CartesianGrid,Legend,ResponsiveContainer} from "recharts";
import '../style/GraphsPages.css';

function LineGraph({ data, xKey, yKey, lineColor, title }) {
  const formattedTitle = title.replace(/graph/i, "").trim();

  return (
    <div style={{border: "0.5px solid #555",padding: "10px",borderRadius: "5px",backgroundColor: "#000"}}>
      <h3 style={{ fontFamily: "'Roboto Mono', monospace", color: "#ccc", marginBottom: "10px" }}>
          {formattedTitle}
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} style={{ backgroundColor: "#000" }}>
          <CartesianGrid stroke="#333" strokeDasharray="3 3" />

          <XAxis
            dataKey={xKey}
            tick={false}       // hide all tick marks and labels
            axisLine={{ stroke: '#ccc' }} // show axis line
          />

          <YAxis tick={{ fill: "#ccc" }} stroke="#ccc" />

          <Tooltip
            contentStyle={{
              backgroundColor: "#000",
              border: "0.5px solid #555",
              color: "#fff",
            }}
          />

          <Legend wrapperStyle={{ color: "#ccc" }} />

          <Line
            type="monotone"
            dataKey={yKey}
            stroke={lineColor}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default LineGraph;
