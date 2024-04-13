import React from "react";
import { useData } from "../DataContext";

const MatchResults = () => {
  const { data } = useData();

  if (!data) return <p>No data available</p>;

  return (
    <div className="grid grid-cols-2 gap-4 p-4">
      {data.map((item) => (
        <div key={item.id} className="p-3 border rounded shadow-sm">
            {/* pattern name attribute */}
          <p>
            <span className="font-semibold">Pattern name:</span>{" "}
            {item.pattern_name}
          </p>
          {/* confidence attribute */}
          <p>
            <span className="font-semibold">Confidence:</span>{" "}
            {item.confidence.toFixed(4)}%
          </p>
          <p>
            {/* pattern url attribute */}
            <span className="font-semibold">Pattern URL:</span>{" "}
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
