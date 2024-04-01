import React from "react";

function Landing() {
    return (
      <div>
        <div className="py-4 px-20">
          <h1 className="text-2xl font-merriweather font-semibold">
            Transferware.<span className="text-blue-500">ai</span>
          </h1>
        </div>
        <div className="flex-row p-6">
          <div className="flex-col sm:px-20">
            <div className="lg:w-2/5">
              <h1 className="font-bold text-6xl my-6">
                Upload your sherd, Find your pattern
              </h1>
              <p className="my-6">
                Explain how this tool is used blah blah blah ai blah. Lorem
                ipsum dolor sit amet, consectetur adipiscing elit, sed do
                eiusmod tempor incididunt ut labore et dolore magna.
              </p>
              <button className="bg-black text-white p-4 px-10 rounded-lg rounded-bl-none">
                Use tool
              </button>
            </div>
            <div className="flex-row space-x-20 my-10 font-semibold">
              <span>Upload sherd</span>
              <span>view 10 closest matches</span>
              <span>get info from our database</span>
            </div>
          </div>
        </div>
      </div>
    );
}
export default Landing;