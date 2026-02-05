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
            Real-Time Aircraft Data Monitoring and Incident Analyzer
          </div>
          <div className="title">BlackboxX</div>
          <div className="description">
            BlackBoxX is a real-time flight monitoring and incident analytics
            platform that processes live and post-flight instrument data. It
            uses backend-heavy rule-based analysis (not ML) to ensure speed,
            robustness, and easy interpretation.
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
            <h2>Blackbox Analysis</h2>
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
