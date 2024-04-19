import React from "react";
import { useLocation } from "react-router-dom"; // Import useLocation
import MatchResults from "../components/matchResults";

function ViewMatches() {
  const location = useLocation(); // Retrieve location object
  const { imagePreviewUrl } = location.state || {}; // Destructure imagePreviewUrl from state, defaulting to an empty object if state is undefined

  return (
    <div className="flex flex-col xl:flex-row h-screen">
      {imagePreviewUrl ? (
        <div className="flex items-center px-12 bg-zinc-900">
          <img
            className="max-w-80  shadow-[0_3px_10px_#718096] rounded-sm"
            src={imagePreviewUrl}
            alt="Uploaded Preview"
          />
        </div>
      ) : (
        <p>No image preview available</p>
      )}
      <div className="flex1 justify-center items-center overflow-y-auto">
        <MatchResults />
      </div>
    </div>
  );
}

export default ViewMatches;
