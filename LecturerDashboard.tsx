// src/pages/LecturerDashboard.tsx
import React, { useEffect, useState } from "react";
import { collection, onSnapshot } from "firebase/firestore";
import { db } from "@/firebase";
import { CalendarDays, Users, MapPinned, Flame } from "lucide-react";
import { useAuth } from "@/lib/auth";

type HeatCell = {
  building: string;
  room: string;
  count: number;
};

export default function LecturerDashboard() {
  const { user } = useAuth();

  const [stats, setStats] = useState({
    totalReservations: 0,
    totalSeats: 0,
    rooms: 0,
  });

  const [heatmapData, setHeatmapData] = useState<HeatCell[]>([]);

  useEffect(() => {
    const unsub = onSnapshot(collection(db, "reservations"), (snap) => {
      const reservations = snap.docs.map((d) => d.data() as any);

      const totalReservations = reservations.length;
      const totalSeats = reservations.reduce(
        (sum, r) => sum + (r.seats?.length || 0),
        0
      );
      const uniqueRooms = new Set(reservations.map((r) => r.building)).size;

      setStats({
        totalReservations,
        totalSeats,
        rooms: uniqueRooms,
      });

      const usageByBlock: Record<string, Record<string, number>> = {};

      reservations.forEach((r) => {
        const raw = String(r.building || "");
        const parts = raw.split("•").map((p: string) => p.trim());

        const roomCode = parts[0] || "Unknown room";
        let blockLabel = parts[1] || "";

        if (!blockLabel) {
          const prefix = roomCode.charAt(0).toUpperCase();
          blockLabel = prefix ? `Block ${prefix}` : "Unknown Block";
        }

        if (!blockLabel.toLowerCase().startsWith("block")) {
          blockLabel = `Block ${blockLabel}`;
        }

        if (!usageByBlock[blockLabel]) usageByBlock[blockLabel] = {};
        usageByBlock[blockLabel][roomCode] =
          (usageByBlock[blockLabel][roomCode] || 0) + 1;
      });

      const formatted: HeatCell[] = [];
      Object.entries(usageByBlock).forEach(([block, roomMap]) => {
        Object.entries(roomMap).forEach(([room, count]) => {
          formatted.push({ building: block, room, count });
        });
      });

      setHeatmapData(formatted);
    });

    return () => unsub();
  }, []);

  const grouped = groupByBuilding(heatmapData);

  const getColor = (count: number) => {
    if (count <= 4) return "bg-green-600";
    if (count <= 8) return "bg-yellow-500";
    return "bg-red-600";
  };

  return (
    <div className="mx-auto max-w-6xl space-y-10 p-6 text-slate-900 dark:text-white">
      {/* Header */}
      <header>
        <h1 className="text-3xl font-bold text-blue-600 dark:text-blue-300 drop-shadow-sm">
          Lecturer Dashboard
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
          Welcome back, {user?.email}. Here is your teaching-space overview.
        </p>
      </header>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          icon={
            <CalendarDays className="h-5 w-5 text-blue-500 dark:text-blue-300" />
          }
          label="Total Reservations"
          value={stats.totalReservations}
        />
        <StatCard
          icon={
            <Users className="h-5 w-5 text-emerald-500 dark:text-emerald-300" />
          }
          label="Seats Booked"
          value={stats.totalSeats}
        />
        <StatCard
          icon={
            <MapPinned className="h-5 w-5 text-amber-500 dark:text-amber-300" />
          }
          label="Rooms Used"
          value={stats.rooms}
        />
      </div>

      {/* Heatmap */}
      <section className="space-y-6">
        <h2 className="text-2xl font-semibold text-blue-600 dark:text-blue-200 flex items-center gap-2">
          <Flame className="h-5 w-5 text-red-400" />
          Real-Time Classroom Heatmap
        </h2>

        {grouped.map((group) => (
          <div key={group.building} className="space-y-3">
            <h3 className="text-xl font-semibold text-blue-600 dark:text-blue-200 flex items-center gap-2">
              <MapPinned className="h-5 w-5 text-blue-300" />
              {group.building}
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {group.rooms
                .sort((a, b) => a.room.localeCompare(b.room))
                .map((room) => (
                  <div
                    key={room.room}
                    className="glass card p-4 text-center relative overflow-hidden"
                  >
                    <div
                      className={`${getColor(
                        room.count
                      )} absolute inset-0 opacity-30`}
                    />
                    <div className="relative z-10">
                      <p className="font-semibold">{room.room}</p>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        {room.count} bookings
                      </p>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        ))}

        {/* Legend */}
        <div className="glass card p-4 text-sm mt-8">
          <p classname="font-semibold text-blue-600 dark:text-blue-300 mb-2 flex items-center gap-2">
            <Flame className="h-4 w-4 text-red-400" />
            Heatmap Legend
          </p>

          <ul className="space-y-1 text-slate-700 dark:text-slate-300">
            <li className="flex items-center gap-2">
              <span className="w-4 h-4 bg-green-600 rounded" /> Low Usage (0–4)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-4 h-4 bg-yellow-500 rounded" /> Medium Usage
              (5–8)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-4 h-4 bg-red-600 rounded" /> High Usage (9+)
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}

function StatCard({ icon, label, value }) {
  return (
    <div className="glass card flex flex-col gap-2">
      <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400 text-sm">
        {icon}
        {label}
      </div>
      <div className="text-3xl font-bold text-blue-600 dark:text-blue-300">
        {value}
      </div>
    </div>
  );
}

function groupByBuilding(data: HeatCell[]) {
  const map: Record<string, HeatCell[]> = {};
  data.forEach((item) => {
    if (!map[item.building]) map[item.building] = [];
    map[item.building].push(item);
  });
  return Object.entries(map).map(([building, rooms]) => ({ building, rooms }));
}
