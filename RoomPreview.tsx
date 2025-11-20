// src/components/RoomPreview.tsx
import React from "react";

export default function RoomPreview({
  layout,
}: {
  layout: "square" | "round" | "lecture" | "none";
}) {
  if (layout === "none") return null;
  const cols = layout === "lecture" ? 20 : 6;
  const rows = layout === "lecture" ? 10 : 6;
  return (
    <div className="rounded-xl border p-4">
      <p className="text-sm text-slate-600 mb-2">
        Preview: <b>{layout}</b>
      </p>
      <div
        className="grid gap-1"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0,1fr))` }}
      >
        {Array.from({ length: rows * cols }).map((_, i) => (
          <div key={i} className="h-3 rounded bg-slate-200" />
        ))}
      </div>
    </div>
  );
}
