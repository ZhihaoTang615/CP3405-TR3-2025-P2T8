// src/pages/AdminReservations.tsx
import React, { useEffect, useState } from "react";
import { collection, onSnapshot, orderBy, query } from "firebase/firestore";
import { db } from "@/firebase";
import { Calendar, MapPin, User2 } from "lucide-react";

interface ReservationRow {
  id: string;
  uid?: string;
  name?: string;
  email?: string;
  building?: string;
  course?: string;
  seats?: string[];
  date: string;
}

export default function AdminReservations() {
  const [items, setItems] = useState<ReservationRow[]>([]);

  useEffect(() => {
    const q = query(collection(db, "reservations"), orderBy("date", "desc"));
    const unsub = onSnapshot(q, (snap) => {
      const data = snap.docs.map((d) => ({
        id: d.id,
        ...(d.data() as any),
      }));
      setItems(data as ReservationRow[]);
    });
    return () => unsub();
  }, []);

  return (
    <div className="mx-auto max-w-5xl p-6 space-y-6 text-[var(--text)] route-fade">
      <header className="space-y-1">
        <h1 className="text-3xl font-bold flex items-center gap-2 text-[var(--accent)]">
          All Reservations (Admin)
        </h1>
        <p className="text-sm text-[var(--muted)]">
          Full list of SmartSeat reservations for auditing and monitoring.
        </p>
      </header>

      {items.length === 0 ? (
        <div className="card glass text-center text-[var(--muted)] py-6">
          No reservations found.
        </div>
      ) : (
        <div className="space-y-4">
          {items.map((it) => (
            <div key={it.id} className="card glass text-sm space-y-1">
              <div className="flex items-center gap-2 text-[var(--muted)]">
                <Calendar className="h-4 w-4 text-[var(--accent)]" />
                <span>{new Date(it.date).toLocaleString()}</span>
              </div>

              <div className="flex items-center gap-2 text-[var(--text)] font-medium">
                <MapPin className="h-4 w-4 text-blue-400" />
                <span>
                  {it.building || "Unknown room"} · {it.course || "Course"}
                </span>
              </div>

              <div className="flex items-center gap-2 text-[var(--muted)]">
                <User2 className="h-4 w-4 text-emerald-400" />
                <span>
                  {it.name || "Student"} ({it.email || "No email"})
                </span>
              </div>

              <div className="text-[var(--muted)]">
                Seats: {it.seats?.join(", ") || "—"}
              </div>

              {it.uid && (
                <div className="text-[10px] text-[var(--muted)] opacity-70">
                  uid: <code>{it.uid}</code>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
