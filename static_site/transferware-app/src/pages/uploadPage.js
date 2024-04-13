import React, { useState, useRef} from "react";
import { useNavigate } from 'react-router-dom'; 
import photoIcon from "../assets/images/photo-icon.png";
import cross from "../assets/images/Cross.png";
import { useData } from "../DataContext";

function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [fileSize, setFileSize] = useState("");
  const fileInputRef = useRef(null); // Added useRef to create a reference to the file input
  const { setData } = useData();
  const navigate = useNavigate(); 

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && (file.type === "image/png" || file.type === "image/jpeg")) {
      setSelectedFile(file);
      setErrorMessage("");
      setUploadedFileName(file.name);

      // Calculate file size in kilobytes
      const fileSizeInKB = Math.round((file.size / 1024) * 100) / 100;
      setFileSize(fileSizeInKB + " KB");
    } else {
      setSelectedFile(null);
      setUploadedFileName(""); // Clear uploaded file name
      setFileSize(""); // Clear file size
      setErrorMessage("Please select a valid PNG or JPG file.");
    }
  };

  const handleUrlChange = (e) => {
    setImageUrl(e.target.value);
  };

  const handleImportFromUrl = () => {
    if (imageUrl.trim() !== "") {
      // Create a new Image object
      const img = new Image();
      img.src = imageUrl;

      // Once the image is loaded, update the UI with the image
      img.onload = () => {
        setSelectedFile(null);
        setErrorMessage("");
        setUploadedFileName("");
        setErrorMessage("");

        // Extract filename from URL and remove query parameters
        let filename = imageUrl.substring(imageUrl.lastIndexOf("/") + 1);
        const queryIndex = filename.indexOf("?");
        if (queryIndex !== -1) {
          filename = filename.substring(0, queryIndex);
        }

        // Update the UI to display the filename
        setImageUrl("");
        setUploadedFileName(filename);
      };

      // Error message if loading fails
      img.onerror = () => {
        setErrorMessage("Error: Unable to load image from URL.");
      };
    } else {
      // If the URL is empty, display an error message
      setErrorMessage("Please enter a valid image URL.");
    }
  };

  const handleCancel = () => {
    setSelectedFile(null);
    setUploadedFileName("");
    setFileSize("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

const handleSubmit = async () => {
  if (selectedFile) {
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Post the file to the query endpoint
      const response = await fetch("http://0.0.0.0:8080/query", {
        method: "POST",
        body: formData,
      });

      // Parse the JSON response
      const queryResults = await response.json();

      // Create a list of fetch promises for each pattern id
      const patternFetchPromises = queryResults.map(async (result) => {
        const patternResponse = await fetch(
          `http://0.0.0.0:8080/pattern/${result.id}`
        );
        const patternData = await patternResponse.json();
        return {
          id: result.id,
          confidence: result.confidence,
          pattern_name: patternData.pattern_name,
          tcc_url: patternData.tcc_url,
        };
      });

      // Wait for all fetch calls to resolve
      const combinedResults = await Promise.all(patternFetchPromises);

      // Fetch images for each pattern with fallback for errors
      const patternImagePromises = combinedResults.map(async (pattern) => {
        try {
          const imageUrlResponse = await fetch(
            `http://0.0.0.0:8080/pattern/image/${pattern.id}`
          );
          if (!imageUrlResponse.ok) throw new Error("Failed to load image");
          const imageUrl = await imageUrlResponse.url; // Assuming the URL itself is what you need
          return {
            ...pattern,
            imageUrl: imageUrl,
          };
        } catch {
          // Fallback image path if the fetch fails or response is not OK
          return {
            ...pattern,
            imageUrl:
              "https://cdn.vectorstock.com/i/500p/36/49/no-image-symbol-missing-available-icon-gallery-vector-43193649.jpg",
          };
        }
      });

      const finalResults = await Promise.all(patternImagePromises);
      setData(finalResults);

      // Navigate to the next page
      navigate("/viewMatches");
    } catch (error) {
      console.error("Error fetching data:", error);
      setErrorMessage("Failed to submit the file.");
    }
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
            ref={fileInputRef}
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
            className="bg-gray-950 text-white px-4 py-2 rounded-md"
          >
            Import
          </button>
        </div>

        {/* uploaded file name & size display */}
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

        <div className="flex justify-center space-x-4">
          <button
            className={`border-2 border-black text-black px-4 py-2 rounded-md ${
              selectedFile ? "" : "opacity-30 cursor-not-allowed"
            }`}
            disabled={!selectedFile}
            onClick={handleCancel}
          >
            Cancel
          </button>
          <button
            className={`bg-blue-400  text-black px-4 py-2 rounded-md ${
              selectedFile ? "" : "opacity-70 cursor-not-allowed"
            }`}
            disabled={!selectedFile}
            onClick={handleSubmit}
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
}

export default UploadPage;
