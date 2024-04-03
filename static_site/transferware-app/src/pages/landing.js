import React from "react";
import group3 from "../assets/images/Group-3.png";
import uploadIcon from "../assets/images/upload-icon.png";
import viewIcon from "../assets/images/view-icon.png";
import databaseIcon from "../assets/images/database-icon.png";


function Landing() {
    return (
      <div className="h-screen overflow-y-hidden relative">
        <div className="py-4 px-10 lg:px-20 h-[64px]">
          <h1 className="text-2xl font-semibold">
            Transferware.<span className="text-blue-500">ai</span>
          </h1>
        </div>
        <div className="flex flex-col lg:flex-row">
          <div className="header-column flex flex-col pb-0 pt-10 px-0 lg:py-16 sm:px-20">
            <div className="header lg:w-full px-10 sm:px-0">
              <h1 className="font-bold text-3xl lg:text-6xl my-6">
                Upload your sherd, Find your pattern
              </h1>
              <p className="my-8 lg:w-4/5">
                Explain how this tool is used blah blah blah ai blah. Lorem
                ipsum dolor sit amet, consectetur adipiscing elit, sed do
                eiusmod tempor incididunt ut labore et dolore magna.
              </p>
              <button className="bg-black font-semibold text-white p-4 px-14 rounded-lg rounded-bl-none">
                Use tool
              </button>
            </div>
            <div className="flex flex-row space-x-5 sm:space-x-10 my-20 font-semibold text-xs px-8 sm:px-0">
              <span className="flex flex-row items-center">
                <img src={uploadIcon} className="h-[20px] px-5" />
                Upload sherd
              </span>
              <span className="flex flex-row items-center">
                <img src={viewIcon} className="h-[20px] px-5" />
                view 10 closest matches
              </span>
              <span className="flex flex-row items-center">
                <img src={databaseIcon} className="h-[20px] px-5" />
                get info from our database
              </span>
            </div>
          </div>
          <div className="flex flex-col justify-end items-end w-full lg:h-[calc(100vh-64px)]">
            <img src={group3} className="w-full" />
          </div>
        </div>
      </div>
    );
}

export default Landing;