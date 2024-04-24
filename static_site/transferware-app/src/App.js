import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { DataProvider } from "./DataContext"; // Import the DataProvider
import "./App.css";
import Navbar from "./components/navbar"; 
import NavbarAlternate from "./components/navbarAlternate"; 
import Landing from "./pages/landing.js";
import UploadPage from "./pages/uploadPage.js";
import ViewMatches from "./pages/viewMatches";
import LoadingAnimation from "./pages/loadingPage";

function App() {
  return (
    <Router>
      <DataProvider>
        <Routes>
          <Route path="/" element={<><Navbar /><Landing /></>} />
          <Route path="/uploadPage" element={<><NavbarAlternate /><UploadPage /></>} />
          <Route path="/viewMatches" element={<><Navbar /><ViewMatches /></>} />
          <Route path="/loading" element={<><NavbarAlternate/><LoadingAnimation /></>} />
        </Routes>
      </DataProvider>
    </Router>
  );
}

export default App;
