import React from "react";
import photoIcon from "../assets/images/photo-icon.png";
function UploadPage() {
  return (
    <div className="flex flex-col items-center h-screen py-20">
      <h1 className="w-full py-10 text-center font-semibold text-xl">
        Upload a photo of your sherd
      </h1>
      <div className="w-5/6 h-1/3 flex flex-col items-center justify-center border-2">
        <img src={photoIcon} className="" alt="photo-icon" />
        <h2 className="font-medium py-4">
          Drop your image here, or
          <span className="font-bold text-blue-600"> browse</span>
        </h2>
        <p className="text-neutral-400 text-xs">
          Supports: PNG, JPG, JPEG, WEBP
        </p>
      </div>
      <div class="relative flex py-5 w-1/2 items-center w-screen">
        <div class="flex-grow border-t border-gray-400"></div>
        <span class="flex-shrink mx-4 text-gray-400">or</span>
        <div class="flex-grow border-t border-gray-400"></div>
      </div>
      <div className="w-5/6">
        <div className="p-4">Import from URL</div>
        <div className="bg-slate-200 w-full h-1/2 rounded-md"></div>
      </div>
    </div>
  );
}

export default UploadPage;
