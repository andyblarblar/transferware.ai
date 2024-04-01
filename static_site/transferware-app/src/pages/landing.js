import React from "react";
import plate1 from "../assets/images/plate1.jpg"
import plate3 from "../assets/images/plate3.jpg";
import plate4 from "../assets/images/plate4.jpg";
import plate11 from "../assets/images/plate11.jpg";
import plate6 from "../assets/images/plate6.jpg";
import plate7 from "../assets/images/plate7.jpg";
import plate12 from "../assets/images/plate12.jpg";
import plate13 from "../assets/images/plate13.jpg";
import group3 from "../assets/images/Group-3.png";

function Landing() {
    return (
      <div className="h-screen overflow-y-hidden">
        <div className="py-4 px-10 sm:px-20 h-[64px]">
          <h1 className="text-2xl font-semibold">
            Transferware.<span className="text-blue-500">ai</span>
          </h1>
        </div>
        <div className="flex flex-col lg:flex-row">
          <div className="header-column flex flex-col pb-0 pt-10 px-16 sm:py-16 sm:px-20">
            <div className="header lg:w-full">
              <h1 className="font-bold text-3xl sm:text-6xl my-6">
                Upload your sherd, Find your pattern
              </h1>
              <p className="my-8 sm:w-4/5">
                Explain how this tool is used blah blah blah ai blah. Lorem
                ipsum dolor sit amet, consectetur adipiscing elit, sed do
                eiusmod tempor incididunt ut labore et dolore magna.
              </p>
              <button className="bg-black text-white p-4 px-14 rounded-lg rounded-bl-none">
                Use tool
              </button>
            </div>
            <div className="flex flex-row space-x-10 my-20 font-semibold">
              <span>Upload sherd</span>
              <span>view 10 closest matches</span>
              <span>get info from our database</span>
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