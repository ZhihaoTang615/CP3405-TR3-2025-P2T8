// src/pages/AdminOverview.tsx
import React, { useEffect, useState } from "react";
import {
  LayoutDashboard,
  Users,
  GraduationCap,
  UserCheck,
  UserCircle2,
  CalendarCheck2,
} from "lucide-react";
import { collection, onSnapshot } from "firebase/firestore";
import { db } from "@/firebase";

type UserDoc = {
  role?: string;
};

export default function AdminOverview() {
  const [userStats, setUserStats] = useState({
    totalUsers: 0,
    students: 0,
    lecturers: 0,
    admins: 0,
  });

  const [totalReservations, setTotalReservations] = useState(0);

  useEffect(() => {
    const unsubUsers = onSnapshot(collection(db, "users"), (snap) => {
      const users = snap.docs.map((d) => d.data() as UserDoc);

      setUserStats({
        totalUsers: users.length,
        students: users.filter((u) => u.role === "student").length,
        lecturers: users.filter((u) => u.role === "lecturer").length,
        admins: users.filter((u) => u.role === "admin").length,
      });
    });

    const unsubRes = onSnapshot(collection(db, "reservations"), (snap) =>
      setTotalReservations(snap.docs.length)
    );

    return () => {
      unsubUsers();
      unsubRes();
    };
  }, []);

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-10 route-fade text-[var(--text)]">
      {/* ===== HEADER ===== */}
      <header className="space-y-1">
        <h1 className="text-3xl font-bold flex items-center gap-2 text-[var(--accent)]">
          <LayoutDashboard className="h-7 w-7 text-[var(--accent)]" />
          Admin Overview
        </h1>
        <p className="text-sm text-[var(--muted)]">
          Campus overview of users & reservation activity.
        </p>
      </header>

      {/* ===== USER STAT CARDS ===== */}
      <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          label="Total Users"
          value={userStats.totalUsers}
          icon={<Users className="h-5 w-5 text-blue-400" />}
        />
        <StatCard
          label="Students"
          value={userStats.students}
          icon={<GraduationCap className="h-5 w-5 text-emerald-400" />}
        />
        <StatCard
          label="Lecturers"
          value={userStats.lecturers}
          icon={<UserCheck className="h-5 w-5 text-amber-400" />}
        />
        <StatCard
          label="Admins"
          value={userStats.admins}
          icon={<UserCircle2 className="h-5 w-5 text-pink-400" />}
        />
      </section>

      {/* ===== RESERVATION SUMMARY ===== */}
      <section className="card p-6 space-y-4">
        <h2 className="text-xl font-semibold flex items-center gap-2 text-[var(--accent)]">
          <CalendarCheck2 className="h-6 w-6 text-[var(--accent)]" />
          Reservation Summary
        </h2>

        <div className="flex items-baseline gap-2">
          <span className="text-6xl font-bold text-[var(--accent)]">
            {totalReservations}
          </span>
          <span className="text-sm text-[var(--muted)]">
            total reservations
          </span>
        </div>

        <ul className="text-sm space-y-1 text-[var(--muted)]">
          <li>• Includes lecturer and student reservations.</li>
          <li>• Visit Users / Reservations pages for detailed overview.</li>
        </ul>
      </section>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
}) {
  return (
    <div className="card p-4 flex flex-col gap-2">
      <div className="flex items-center gap-2 text-sm">
        {icon}
        <span>{label}</span>
      </div>
      <div className="text-4xl font-bold text-[var(--accent)]">{value}</div>
    </div>
  );
}
