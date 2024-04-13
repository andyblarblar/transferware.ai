import React from "react";
import { useData } from "../DataContext";

const MatchResults = () => {
  const { data } = useData();

  if (!data) return <p>No data available</p>;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 p-4 w-5/6">
      {data.map((item) => (
        <div key={item.id} className="p-3 border rounded shadow-sm">
          <img
            src={item.imageUrl}
            alt="Pattern-img"
            className="w-1/2 lg:w-2/5 h-auto mb-2"
          />
          <p className="mb-4">
            <span className="font-semibold">Pattern name:</span>{" "}
            {item.pattern_name}
          </p>
          <p className="mb-4">
            <span className="font-semibold">Confidence:</span>{" "}
            {item.confidence.toFixed(3)}%
          </p>
          <p className="break-words break-all mb-2">
            <span className="font-semibold">Pattern URL: </span>
            <a
              className="text-blue-600"
              href={item.tcc_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              {item.tcc_url}
            </a>
          </p>
        </div>
      ))}
    </div>
  );
};

export default MatchResults;

