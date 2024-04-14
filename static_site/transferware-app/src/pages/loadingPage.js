import React from "react";
import "./loadingPage.css"
import archeologistDigging from "../assets/gifs/archeologist-digging.gif"
function LoadingAnimation(){
    return (
      <div className="flex bg-amber-50 flex-col justify-center items-center h-screen">
        <div className="flex flex-row justify-center items-center pt-16">
          <img className="h-full" src={archeologistDigging}></img>
          <div class="loader"></div>
          <div class="loader2"></div>
        </div>

        <div className="w-screen bg-amber-100 h-full"></div>
      </div>
    );
}
export default LoadingAnimation;
