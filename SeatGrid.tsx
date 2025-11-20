import React from "react";

export type Seat = {
  id: string; // e.g. "A1"
  status: "free" | "reserved" | "yours";
};

type Props = {
  seats: Seat[]; // 8x8 或任意
  onPick: (id: string) => void;
  selectedId: string | null;
};

export default function SeatGrid({ seats, onPick, selectedId }: Props) {
  return (
    <div className="grid grid-cols-8 gap-3">
      {seats.map((s) => {
        const isSelected = selectedId === s.id;
        const base =
          s.status === "reserved"
            ? "bg-amber-100 text-amber-700 cursor-not-allowed"
            : s.status === "yours"
            ? "bg-indigo-100 text-indigo-700 ring-1 ring-indigo-300"
            : "bg-green-100 text-green-700 hover:bg-green-200";
        const selectedRing = isSelected ? " ring-2 ring-indigo-500" : "";
        return (
          <button
            key={s.id}
            disabled={s.status === "reserved"}
            onClick={() => onPick(s.id)}
            className={
              "h-10 rounded-md text-sm font-medium transition " +
              base +
              selectedRing
            }
            title={s.id}
          >
            {s.id}
          </button>
        );
      })}
    </div>
  );
}
