import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <div className=" flex flex-row items-center justify-between shadow-md w-full fixed py-3 px-6 lg:px-20 ">
      <h1 className="text-2xl font-semibold">
        Transferware.<span className="text-blue-500">ai</span>
      </h1>
      <ul className="flex font-semibold space-x-2 sm:space-x-8 sm:text-lg">
        <li className="hover:text-blue-500">
          <Link to="/">Home</Link>
        </li>
        <li className="hover:text-blue-500">
          <Link to="/">About</Link>
        </li>
        <li className="hover:text-blue-500">
          <Link to="/uploadPage">Upload</Link>
        </li>
      </ul>
    </div>
  );
};

export default Navbar;
