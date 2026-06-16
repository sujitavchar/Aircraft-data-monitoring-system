import React from "react";
import { useNavigate } from "react-router-dom";
import '../style/LandingPage.css';

function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-container">
      {/* Hero Section */}
      <section className="hero">
        <div>
          <div className="tagline">
           Aircraft Data Monitoring & Analysis System
          </div>
          <div className="title">Aerion</div>
          <div className="description">
            Aerion is a software system designed to analyse aircraft telemetry across two critical phases: live flight
          monitoring and post-incident black box investigation. Its core purpose is to compress the gap between
          data generation and actionable insight, enabling faster anomaly detection and a meaningful shift from
          reactive to proactive aviation safety.

          </div>
          <button
            className="cta"
            onClick={() =>
              document
                .getElementById("analysis-sections")
                .scrollIntoView({ behavior: "smooth" })
            }
          >
            Learn More <span className="icon">✈</span>
          </button>
        </div>
      </section>

      {/* Analysis Section */}
      <section id="analysis-sections">
        <div className="analysis-container-side">
          
          {/* Left Card */}
          <div className="analysis-card left">
            <h2>Realtime Analysis</h2>
            <p>
              Monitor live flights and instrument data with instant insights and
              alerts. Our platform provides actionable analytics in real time
              for better decision-making.
            </p>
            <button
              className="cta-button"
              onClick={() => navigate("/livetracking")}
            >
              Take-Off <span className="icon">✈</span>
            </button>
          </div>

          {/* Divider */}
          <div className="vertical-divider"></div>

          {/* Right Card */}
          <div className="analysis-card right">
            <h2>Blackbox (FDR) Analysis</h2>
            <p>
              Post-flight data analysis to detect anomalies and optimize
              performance. Detailed rule-based reports ensure fast, reliable,
              and interpretable results.
            </p>
            <button
              className="cta-button"
              onClick={() => navigate("/upload")}
            >
              Start <span className="icon">✈</span>
            </button>
          </div>
          
        </div>
      </section>
    </div>
  );
}

export default LandingPage;
