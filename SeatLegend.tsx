import React from "react";

export default function SeatLegend() {
  return (
    <div className="flex items-center gap-4 text-xs text-slate-600">
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-3 w-5 rounded bg-green-100 ring-1 ring-green-300" />
        Free
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-3 w-5 rounded bg-indigo-100 ring-1 ring-indigo-300" />
        Yours
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-3 w-5 rounded bg-amber-100 ring-1 ring-amber-300" />
        Reserved
      </span>
    </div>
  );
}
