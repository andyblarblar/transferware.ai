import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";
import Landing from "./pages/landing.js";
import UploadPage from "./pages/uploadPage.js"

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/uploadPage" element={<UploadPage />} />
      </Routes>
    </Router>
  );
}

export default App;
