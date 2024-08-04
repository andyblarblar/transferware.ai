import React from "react";
import "./loadingPage.css";
import underline1 from "../assets/images/zigzag-underline.png";
import plate1 from "../assets/images/plate1.jpg"
import plate2 from "../assets/images/plate2.jpg"
import plate3 from "../assets/images/plate3.png";
import Footer from "../components/footer";
import lightIcon from "../assets/images/Light-On.png"
import focusIcon from "../assets/images/focus-icon.png"
import cleanIcon from "../assets/images/sheet-icon.png"
function AboutPage() {
  return (
    <div className="flex flex-col items-center  pt-[52px]">
      <div className="bg-custom-image bg-cover bg-center h-64 w-screen shadow-md relative">
        <div className="absolute inset-0 flex items-center justify-center">
          {/* <h1 className="text-black text-5xl font-bold">About us</h1> */}
        </div>
      </div>
      <div>
        <div className="flex flex-col items-center">
          <h1 className="lg:text-2xl sm: text-xl font-semibold pt-9 pb-3">
            Find a Match to Your Sherds
          </h1>
          <img src={underline1} className="lg:w-2/12 w-1/3" />
        </div>
        <div className="w-screen lg:px-24">
          <div className="flex flex-row items-center justify-center md:space-x-14">
            <div className="flex flex-col w-1/3">
              <h1 className="font-bold text-1xl">Simplify Your Research</h1>
              <p className="font-light">
                With Transferware.ai, you can effortlessly identify and date
                transferware sherds by matching your images to a comprehensive
                database of patterns. Our goal is to make your research faster
                and more efficient, allowing you to focus on your discoveries
                instead of manual identification.
              </p>
            </div>
            <img src={plate1} className="lg:w-1/6 md:w-1/4 w-1/2" />
          </div>
        </div>

        <div className="w-screen lg:px-24 py-6">
          <div className="flex flex-row items-center justify-center md:space-x-14">
            <img src={plate2} className="lg:w-1/6 md:w-1/4 w-1/2" />
            <div className="flex flex-col w-1/3">
              <h1 className="font-bold text-1xl">How It Works</h1>
              <p className="font-light">
                Transferware.ai streamlines your workflow by allowing you to
                upload an image of your sherd through drag-and-drop, file
                browsing, or URL paste. Upon submission, the system employs
                machine learning technology to process the image and find the
                nearest matches.
              </p>
            </div>
          </div>
        </div>
        <div className="lg:px-24 mb-16 flex justify-center items-center">
          <div className="flex flex-row items-center justify-center md:space-x-14">
            <div className="flex flex-col w-1/3">
              <h1 className="font-bold text-1xl">Matching and Results</h1>
              <p className="font-light">
                Transferware.ai delivers comprehensive results, including
                pattern names, confidence values, and direct links to the
                Transferware Collectors Club (TCC) database for further
                reference and detailed information.
              </p>
            </div>
            <img src={plate3} className="lg:w-1/6 md:w-1/4 w-1/2" />
          </div>
        </div>
        <div className="lg:px-24 flex flex-col items-center mb-16">
          <h1 className="lg:text-2xl sm: text-xl font-semibold pb-3">
            Best Practices for Uploading Images
          </h1>
          <img src={underline1} className="lg:w-2/12 w-1/3" />
          <div className="flex flex-row justify-center mt-6 space-x-3">
            <div className="flex flex-col items-center md:w-1/5 w-1/4">
              <img src={lightIcon} className="w-1/6" />
              <h1 className="font-bold text-1xl">Lighting</h1>
              <p className="font-light text-center">
                Ensure the image is well-lit and free from shadows. Natural
                light is ideal.
              </p>
            </div>
            <div className="flex flex-col items-center md:w-1/5 w-1/4">
              <img src={focusIcon} className="w-1/6" />
              <h1 className="font-bold text-1xl ">Focus</h1>
              <p className="font-light text-center">
                The sherd should be in clear focus with no blurriness. Focusing
                on the center pattern of the sherd often yields better results.
              </p>
            </div>
            <div className="flex flex-col items-center w-1/5">
              <img src={cleanIcon} className="w-1/5" />
              <h1 className="font-bold text-1xl">Background</h1>
              <p className="font-light text-center">
                Place the sherd on a plain, contrasting background to avoid
                distractions and errors.
              </p>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
export default AboutPage;
