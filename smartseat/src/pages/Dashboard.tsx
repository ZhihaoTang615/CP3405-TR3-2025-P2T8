// src/pages/Dashboard.tsx
import React, { useMemo } from "react";
import { getReservations } from "@/lib/storage";

export default function Dashboard() {
  const data = getReservations();

  const stats = useMemo(() => {
    const total = data.length;
    const seats = data.reduce((acc, r) => acc + r.seats.length, 0);
    const byBuilding = new Map<string, number>();
    data.forEach((r) =>
      byBuilding.set(r.building, (byBuilding.get(r.building) || 0) + 1)
    );
    return { total, seats, byBuilding: Array.from(byBuilding.entries()) };
  }, [data]);

  return (
    <div className="max-w-3xl mx-auto">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-indigo-700">Dashboard</h1>
        <p className="text-slate-600 mt-2">
          Quick stats for your SmartSeat activity.
        </p>
      </header>

      <div className="grid md:grid-cols-3 gap-4">
        <Card title="Total Reservations" value={stats.total} />
        <Card title="Seats Booked" value={stats.seats} />
        <Card title="Buildings Used" value={stats.byBuilding.length} />
      </div>

      <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="text-lg font-semibold mb-4">By Building</h2>
        {stats.byBuilding.length === 0 ? (
          <div className="text-slate-500">No data yet.</div>
        ) : (
          <ul className="space-y-2">
            {stats.byBuilding.map(([name, count]) => (
              <li key={name} className="flex justify-between text-slate-700">
                <span>{name}</span>
                <span className="font-medium">{count}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function Card({ title, value }: { title: string; value: number }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 text-center">
      <div className="text-sm text-slate-500">{title}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-800">{value}</div>
    </div>
  );
}
