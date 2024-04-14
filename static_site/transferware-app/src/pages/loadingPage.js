import React from "react";
import "./loadingPage.css"
import archeologistDigging from "../assets/gifs/archeologist-digging.gif"
function LoadingAnimation(){
    return (
      <div className="flex bg-white flex-col justify-center items-center h-screen">
        <img className="h-3/5" src={archeologistDigging}></img>
        <div class="loader"></div>
        <div class="loader2"></div> 
      </div>
    );
}
export default LoadingAnimation;
