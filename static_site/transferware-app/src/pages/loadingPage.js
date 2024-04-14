import React from "react";
import "./loadingPage.css"
import archeologistDigging from "../assets/gifs/archeologist-digging.gif"
function LoadingAnimation(){
    return (
      <div className="flex bg-neutral-300 flex-col justify-start items-center h-screen">
        <img src={archeologistDigging}></img>
        <div class="loader"></div>
      </div>
    );
}
export default LoadingAnimation;
