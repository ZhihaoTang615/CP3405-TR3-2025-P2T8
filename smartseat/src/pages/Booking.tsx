import React, { useState } from "react";

const Booking: React.FC = () => {
  const [form, setForm] = useState({
    date: "",
    start: "",
    duration: 2,
    building: "Main",
    seatType: "Near Window",
    noise: 40,
    note: "",
  });

  const update = (k: string, v: any) => setForm((s) => ({ ...s, [k]: v }));
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    alert(
      `✅ Booking placed:\nDate: ${form.date}\nStart: ${
        form.start
      }\nDuration: ${form.duration}h\nBuilding: ${form.building}\nSeat Type: ${
        form.seatType
      }\nNoise: ${form.noise} dB\nNote: ${form.note || "-"}`
    );
  };

  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <section className="mx-auto my-10 text-center">
          <h1 className="bg-gradient-to-r from-slate-900 to-indigo-700 bg-clip-text text-3xl font-bold tracking-tight text-transparent sm:text-4xl">
            🪑 Smart Booking
          </h1>
          <p className="mt-3 text-slate-600">
            Choose your preferred date, time, and environment — SmartSeat will
            find your best match.
          </p>
        </section>

        <form
          onSubmit={submit}
          className="grid gap-6 md:grid-cols-2 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
        >
          <div className="grid gap-5">
            <Field label="Date">
              <input
                required
                type="date"
                value={form.date}
                onChange={(e) => update("date", e.target.value)}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 outline-none focus:border-indigo-500 transition"
              />
            </Field>
            <Field label="Start Time">
              <input
                required
                type="time"
                value={form.start}
                onChange={(e) => update("start", e.target.value)}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 outline-none focus:border-indigo-500 transition"
              />
            </Field>
            <Field label={`Duration: ${form.duration} hours`}>
              <input
                type="range"
                min={1}
                max={6}
                value={form.duration}
                onChange={(e) => update("duration", Number(e.target.value))}
                className="w-full accent-indigo-600"
              />
            </Field>
          </div>

          <div className="grid gap-5">
            <Field label="Building">
              <select
                value={form.building}
                onChange={(e) => update("building", e.target.value)}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 outline-none focus:border-indigo-500 transition"
              >
                <option>Main</option>
                <option>Library</option>
                <option>Innovation Hub</option>
              </select>
            </Field>
            <Field label="Seat Type">
              <select
                value={form.seatType}
                onChange={(e) => update("seatType", e.target.value)}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 outline-none focus:border-indigo-500 transition"
              >
                <option>Near Window</option>
                <option>Quiet Corner</option>
                <option>Group Table</option>
                <option>Power Socket</option>
              </select>
            </Field>
            <Field label={`Noise tolerance: ${form.noise} dB`}>
              <input
                type="range"
                min={20}
                max={80}
                step={5}
                value={form.noise}
                onChange={(e) => update("noise", Number(e.target.value))}
                className="w-full accent-indigo-600"
              />
            </Field>
          </div>

          <div className="md:col-span-2">
            <Field label="Note (optional)">
              <textarea
                rows={3}
                value={form.note}
                onChange={(e) => update("note", e.target.value)}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 outline-none focus:border-indigo-500 transition"
                placeholder="e.g., near friends / avoid AC"
              />
            </Field>
          </div>

          <div className="md:col-span-2 flex justify-end">
            <button
              type="submit"
              className="rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg transition hover:-translate-y-0.5 hover:bg-indigo-700 hover:shadow-xl"
            >
              Confirm Booking
            </button>
          </div>
        </form>
      </main>
    </div>
  );
};

const Field: React.FC<{ label: string; children: React.ReactNode }> = ({
  label,
  children,
}) => (
  <label className="block">
    <div className="mb-1.5 text-sm font-medium text-slate-700">{label}</div>
    {children}
  </label>
);

export default Booking;
