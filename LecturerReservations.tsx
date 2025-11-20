// src/pages/LecturerReservations.tsx
import React, { useEffect, useState } from "react";
import { collection, onSnapshot } from "firebase/firestore";
import { db } from "@/firebase";
import { useAuth } from "@/lib/auth";
import { Calendar, Clock, User, MapPin, BookOpen } from "lucide-react";

type Reservation = {
  building: string;
  course?: string;
  email?: string;
  date: string;
  durationHours: number;
  seats?: string[];
  fromRole: string;
};

export default function LecturerReservations() {
  // 这里暂时没用到 user，用下划线避免报警告
  const { user: _user } = useAuth();
  const [lecturerRes, setLecturerRes] = useState<Reservation[]>([]);
  const [studentRes, setStudentRes] = useState<Reservation[]>([]);

  useEffect(() => {
    const unsub = onSnapshot(collection(db, "reservations"), (snap) => {
      const all = snap.docs.map((d) => d.data() as Reservation);

      // 讲师预定：按时间排序（升序）
      const lecturers = all
        .filter((r) => r.fromRole === "lecturer")
        .sort(
          (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
        );

      // 学生预定：按 building 排序（Block + Room）
      const students = all
        .filter((r) => r.fromRole !== "lecturer")
        .sort((a, b) => {
          const A = a.building || "";
          const B = b.building || "";
          return A.localeCompare(B);
        });

      setLecturerRes(lecturers);
      setStudentRes(students);
    });

    return () => unsub();
  }, []);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-12 route-fade text-slate-900 dark:text-white">
      {/* ===== HEADER ===== */}
      <header>
        <h1 className="text-3xl font-bold text-blue-500">All Reservations</h1>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          Manage all lecturer and student bookings.
        </p>
      </header>

      {/* ===================== LECTURER RESERVATIONS ===================== */}
      <section className="space-y-4">
        <h2 className="text-2xl font-semibold flex items-center gap-2 text-blue-500 dark:text-blue-200">
          👨‍🏫 Lecturer Reservations
        </h2>

        {lecturerRes.length === 0 ? (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            No lecturer reservations found.
          </p>
        ) : (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {lecturerRes.map((r, i) => (
              <div key={i} className="glass card p-4">
                {/* Course type */}
                <p className="flex items-center gap-2 font-medium text-sm text-blue-500 dark:text-blue-300">
                  <BookOpen className="h-4 w-4" />
                  {r.course || "Teaching Session"}
                </p>

                {/* Lecturer email */}
                <p className="flex items-center gap-2 mt-2 text-sm text-slate-700 dark:text-slate-300">
                  <User className="h-4 w-4 text-blue-400 dark:text-blue-300" />
                  {r.email || "Unknown lecturer"}
                </p>

                {/* Room / Building */}
                <p className="flex items-center gap-2 mt-1 text-sm text-slate-700 dark:text-slate-300">
                  <MapPin className="h-4 w-4 text-blue-400 dark:text-blue-300" />
                  {r.building}
                </p>

                {/* Date & Time */}
                <p className="flex items-center gap-2 mt-1 text-sm text-slate-700 dark:text-slate-300">
                  <Calendar className="h-4 w-4 text-blue-400 dark:text-blue-300" />
                  {new Date(r.date).toLocaleString()}
                </p>

                {/* Duration */}
                <p className="flex items-center gap-2 mt-1 text-sm text-slate-700 dark:text-slate-300">
                  <Clock className="h-4 w-4 text-blue-400 dark:text-blue-300" />
                  {r.durationHours} hours
                </p>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ===================== STUDENT RESERVATIONS ===================== */}
      <section className="space-y-4">
        <h2 className="text-2xl font-semibold flex items-center gap-2 text-blue-500 dark:text-blue-200">
          🎒 Student Reservations
        </h2>

        {studentRes.length === 0 ? (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            No student reservations available.
          </p>
        ) : (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {studentRes.map((r, i) => (
              <div key={i} className="glass card p-4">
                {/* Course type / fallback */}
                <p className="flex items-center gap-2 font-medium text-sm text-blue-500 dark:text-blue-300">
                  <BookOpen className="h-4 w-4" />
                  {r.course || "Room Reservation"}
                </p>

                {/* Room / Building */}
                <p className="flex items-center gap-2 mt-2 text-sm text-slate-700 dark:text-slate-300">
                  <MapPin className="h-4 w-4 text-blue-400 dark:text-blue-300" />
                  {r.building}
                </p>

                {/* Date & Time */}
                <p className="flex items-center gap-2 mt-1 text-sm text-slate-700 dark:text-slate-300">
                  <Calendar className="h-4 w-4 text-blue-400 dark:text-blue-300" />
                  {new Date(r.date).toLocaleString()}
                </p>

                {/* Seats */}
                <p className="flex items-center gap-2 mt-2 text-sm text-emerald-600 dark:text-emerald-400">
                  💺 {r.seats?.length ?? 0} seats booked
                </p>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
