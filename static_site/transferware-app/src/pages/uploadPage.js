import React, { useState } from "react";
import photoIcon from "../assets/images/photo-icon.png";
import cross from "../assets/images/Cross.png";


function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [imageUrl, setImageUrl] = useState(""); // State to store image URL
  const [fileSize, setFileSize] = useState(""); // State to store image file size

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && (file.type === "image/png" || file.type === "image/jpeg")) {
      setSelectedFile(file);
      setErrorMessage("");
      setUploadedFileName(file.name); // Set uploaded file name

      // Calculate file size in kilobytes
      const fileSizeInKB = Math.round((file.size / 1024) * 100) / 100; // Round to two decimal places
      setFileSize(fileSizeInKB + " KB");
    } else {
      setSelectedFile(null);
      setUploadedFileName(""); // Clear uploaded file name
      setFileSize(""); // Clear file size
      setErrorMessage("Please select a valid PNG or JPG file."); // Set error message
    }
  };

  const handleUrlChange = (e) => {
    setImageUrl(e.target.value);
  };

  const handleImportFromUrl = () => {
    // Check if imageUrl is not empty
    if (imageUrl.trim() !== "") {
      // Create a new Image object
      const img = new Image();
      img.src = imageUrl;

      // Once the image is loaded, update the UI with the image
      img.onload = () => {
        setSelectedFile(null);
        setErrorMessage("");
        setUploadedFileName(""); // Clear uploaded file name
        setErrorMessage("");

        // Extract filename from URL and remove query parameters
        let filename = imageUrl.substring(imageUrl.lastIndexOf("/") + 1);
        const queryIndex = filename.indexOf("?");
        if (queryIndex !== -1) {
          filename = filename.substring(0, queryIndex);
        }

        // Update the UI to display the filename
        setImageUrl("");
        setUploadedFileName(filename); // Set filename as the uploaded file name
      };

      // If there's an error loading the image
      img.onerror = () => {
        // Display an error message
        setErrorMessage("Error: Unable to load image from URL.");
      };
    } else {
      // If the URL is empty, display an error message
      setErrorMessage("Please enter a valid image URL.");
    }
  };


  return (
    <div className="flex flex-col items-center h-screen py-20">
      <h1 className="w-full py-10 text-center font-semibold text-xl">
        Upload a photo of your sherd
      </h1>
      <div className="w-5/6 h-1/3 flex flex-col items-center justify-center border-2">
        <img src={photoIcon} className="" alt="photo-icon" />
        <h2 className="font-medium py-4">
          Drop your image here, or
          <label
            htmlFor="fileInput"
            className="cursor-pointer font-bold text-blue-600"
          >
            {" "}
            browse
          </label>
          <input
            id="fileInput"
            type="file"
            accept=".png,.jpg,.jpeg"
            onChange={handleFileChange}
            className="hidden"
          />
        </h2>
        <p className="text-neutral-400 text-xs">Supports: PNG and JPG images</p>
        {errorMessage && <p className="text-red-500">{errorMessage}</p>}
      </div>
      <div className="relative flex py-5 w-1/2 items-center">
        <div className="flex-grow border-t border-gray-400"></div>
        <span className="flex-shrink mx-4 text-gray-400">or</span>
        <div className="flex-grow border-t border-gray-400"></div>
      </div>
      <div className="w-5/6">
        <div className="p-4">Import from URL</div>
        <div className="flex justify-between p-4 rounded-md">
          <input
            type="text"
            placeholder="Enter image URL"
            value={imageUrl}
            onChange={handleUrlChange}
            className="flex-grow mr-2 border border-gray-950 rounded-md px-2 py-1"
          />
          <button
            onClick={handleImportFromUrl}
            className="bg-blue-400 text-white px-4 py-2 rounded-md"
          >
            Import
          </button>
        </div>
        
        {/* uploaded file name display */}
        <div className="p-4 rounded-md">
          <div className={`w-full rounded-md ${!uploadedFileName && "hidden"}`}>
            {uploadedFileName && (
              <p className="py-2 px-4 w-full flex justify-between">
                {uploadedFileName}
                <span className="flex flex-row text-zinc-400 font-semibold">
                  ({fileSize})
                  <img src={cross} className="" alt="photo-icon" />
                </span>
              </p>
            )}
          </div>
        </div>


      </div>
    </div>
  );
}

export default UploadPage;
