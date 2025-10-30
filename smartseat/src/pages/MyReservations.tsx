// src/pages/MyReservations.tsx
import React, { useEffect, useMemo, useState } from "react";
import { CalendarDays, MapPin, BookOpen, Trash2 } from "lucide-react";
import type { Reservation } from "@/types";
import { getReservations, deleteReservation } from "@/lib/storage";

export default function MyReservations() {
  const [items, setItems] = useState<Reservation[]>([]);

  useEffect(() => {
    setItems(getReservations());
  }, []);

  const remove = (id: number) => {
    deleteReservation(id);
    setItems(getReservations());
  };

  const sorted = useMemo(
    () =>
      [...items].sort(
        (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
      ),
    [items]
  );

  return (
    <div className="max-w-3xl mx-auto">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-indigo-700">My Reservations</h1>
        <p className="text-slate-600 mt-2">
          View and manage your upcoming seat reservations.
        </p>
      </header>

      {sorted.length === 0 ? (
        <div className="rounded-xl border border-slate-200 bg-white p-8 text-center">
          <p className="text-slate-600">You don’t have any reservations yet.</p>
        </div>
      ) : (
        <ul className="space-y-4">
          {sorted.map((r) => {
            const isPast = new Date(r.date).getTime() < Date.now();
            return (
              <li
                key={r.id}
                className="rounded-xl border border-slate-200 bg-white p-5 flex justify-between items-start gap-4"
              >
                <div className="space-y-1">
                  <div className="text-sm text-slate-500">
                    Booked by <span className="font-medium">{r.user.name}</span>{" "}
                    ({r.user.email})
                  </div>
                  <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
                    <div className="inline-flex items-center gap-2">
                      <CalendarDays className="w-4 h-4 text-indigo-600" />
                      <span
                        className={isPast ? "line-through text-slate-400" : ""}
                      >
                        {new Date(r.date).toLocaleString()}
                      </span>
                    </div>
                    <div className="inline-flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-indigo-600" />
                      <span>{r.building}</span>
                    </div>
                    <div className="inline-flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-indigo-600" />
                      <span>{r.course}</span>
                    </div>
                  </div>
                  <div className="text-sm">
                    Seats:{" "}
                    <span className="font-semibold text-indigo-700">
                      {r.seats.join(", ")}
                    </span>
                  </div>
                  {isPast && (
                    <span className="inline-block text-xs mt-1 rounded-full bg-slate-100 text-slate-600 px-2 py-0.5">
                      Past
                    </span>
                  )}
                </div>
                <button
                  onClick={() => remove(r.id)}
                  className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
                  title="Delete"
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
