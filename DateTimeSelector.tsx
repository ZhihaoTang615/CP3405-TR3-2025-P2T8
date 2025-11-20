import React from "react";

type Props = {
  value: string | null; // ISO string
  onChange: (iso: string | null) => void;
};

export default function DateTimeSelector({ value, onChange }: Props) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-slate-700">Date & Time</label>
      <input
        type="datetime-local"
        className="w-full rounded-md border px-3 py-2 text-sm"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        min={new Date().toISOString().slice(0, 16)}
      />
      <p className="text-xs text-slate-500">
        Please select a future time slot.
      </p>
    </div>
  );
}
