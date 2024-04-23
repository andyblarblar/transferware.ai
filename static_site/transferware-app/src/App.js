import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { DataProvider } from "./DataContext"; // Import the DataProvider
import "./App.css";
import Navbar from "./components/navbar"; 
import Landing from "./pages/landing.js";
import UploadPage from "./pages/uploadPage.js";
import ViewMatches from "./pages/viewMatches";
import LoadingAnimation from "./pages/loadingPage";

function App() {
  return (
    <Router>
      <DataProvider>
        <Navbar />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/uploadPage" element={<UploadPage />} />
          <Route path="/viewMatches" element={<ViewMatches />} />
          <Route path="/loading" element={<LoadingAnimation />} />
        </Routes>
      </DataProvider>
    </Router>
  );
}

export default App;
