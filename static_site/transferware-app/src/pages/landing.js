import React from "react";
import plate1 from "../assets/images/plate1.jpg"
import plate3 from "../assets/images/plate3.jpg";
import plate4 from "../assets/images/plate4.jpg";
import plate11 from "../assets/images/plate11.jpg";
import plate6 from "../assets/images/plate6.jpg";
import plate7 from "../assets/images/plate7.jpg";
import plate12 from "../assets/images/plate12.jpg";
import plate13 from "../assets/images/plate13.jpg";


function Landing() {
    return (
      <div className="h-screen">
        <div className="py-4 px-20">
          <h1 className="text-2xl font-semibold">
            Transferware.<span className="text-blue-500">ai</span>
          </h1>
        </div>
        <div className="flex flex-row py-8 px-32 h-full">
          <div className="header-column flex flex-col py-16">
            <div className="header lg:w-3/4">
              <h1 className="font-bold text-6xl my-6">
                Upload your sherd, Find your pattern
              </h1>
              <p className="my-8 w-4/5">
                Explain how this tool is used blah blah blah ai blah. Lorem
                ipsum dolor sit amet, consectetur adipiscing elit, sed do
                eiusmod tempor incididunt ut labore et dolore magna.
              </p>
              <button className="bg-black text-white p-4 px-14 rounded-lg rounded-bl-none">
                Use tool
              </button>
            </div>
            <div className="flex-row space-x-20 my-20 font-semibold">
              <span>Upload sherd</span>
              <span>view 10 closest matches</span>
              <span>get info from our database</span>
            </div>
          </div>
          <div class="grid grid-cols-4 grid-rows-3 gap-4 bg-white text-gray-700 w-3/4 h-3/4">
            <div
              className="box col-span-2 row-span-1 bg-cover bg-center rounded-xl"
              style={{ backgroundImage: `url(${plate1})` }}
            ></div>
            <div
              class="box col-start-3 col-span-2 row-span-1 bg-cover bg-center rounded-xl"
              style={{ backgroundImage: `url(${plate3})` }}
            ></div>
            <div
              class="box col-start-1 row-start-2 row-span-1 bg-cover bg-center rounded-xl"
              style={{ backgroundImage: `url(${plate13})` }}
            ></div>
            <div
              class="box col-start-2 col-span-3 row-start-2 row-span-1 bg-cover bg-center rounded-xl"
              style={{ backgroundImage: `url(${plate6})` }}
            ></div>
            <div
              class="box col-span-4 row-start-3 row-span-1 bg-cover bg-center rounded-xl "
              style={{ backgroundImage: `url(${plate11})` }}
            ></div>
          </div>
        </div>
      </div>
    );
}

export default Landing;