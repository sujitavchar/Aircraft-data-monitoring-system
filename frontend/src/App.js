// File: App.js

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import LandingPage from './component/LandingPage';
import UploadPage from './component/UploadPage';
import LiveTracking from './component/LiveTracking';
import BlackBox from './component/BlackBox';
import GraphsPages from './component/GraphsPages';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/blackbox" element={<BlackBox />} />
        <Route path="/livetracking" element={<LiveTracking />} />
        <Route path="/graphs" element={<GraphsPages />} />
      </Routes>
    </Router>
  );
}

export default App;
