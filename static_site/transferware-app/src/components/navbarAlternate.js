import React from "react";
import { Link } from "react-router-dom";

const NavbarAlternate = () => {
  return (
    <div className=" flex flex-row items-center justify-between text-white w-full fixed py-3 px-6 lg:px-8 ">
      <Link to={"/"} className="text-2xl md:ml-12 font-semibold">
        Transferware.<span className="text-blue-500">ai</span>
      </Link>
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

export default NavbarAlternate;
