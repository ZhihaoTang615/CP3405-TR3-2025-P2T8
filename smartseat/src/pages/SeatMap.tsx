// src/pages/SeatMap.tsx
import React, { useEffect, useMemo, useState } from "react";
import { Building2, BookOpen, Calendar } from "lucide-react";
import type { Seat, SeatStatus, User } from "@/types";
import { addReservation, getReservedSeatIds } from "@/lib/storage";

const SeatMap: React.FC = () => {
  const [page, setPage] = useState<"login" | "signup" | "map" | "reserve">(
    "login"
  );
  const [user, setUser] = useState<User | null>(null);

  // 生成基础座位
  const baseSeats = useMemo<Seat[]>(() => {
    const rows = 6,
      cols = 8;
    const list: Seat[] = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const id = `${String.fromCharCode(65 + r)}${c + 1}`;
        const dice = Math.random();
        const status: SeatStatus = dice < 0.15 ? "reserved" : "free";
        list.push({ id, status });
      }
    }
    return list;
  }, []);

  const [seats, setSeats] = useState<Seat[]>(baseSeats);

  // 从本地存储同步 reserved（刷新后仍然保留）
  useEffect(() => {
    const reservedFromStorage = new Set(getReservedSeatIds());
    if (reservedFromStorage.size === 0) return;
    setSeats((prev) =>
      prev.map((s) =>
        reservedFromStorage.has(s.id) ? { ...s, status: "reserved" } : s
      )
    );
  }, []);

  const toggleSeat = (seat: Seat) => {
    setSeats((prev) =>
      prev.map((it) =>
        it.id === seat.id
          ? {
              ...it,
              status:
                it.status === "mine"
                  ? "free"
                  : it.status === "free"
                  ? "mine"
                  : it.status,
            }
          : it
      )
    );
  };

  const selected = seats.filter((s) => s.status === "mine");

  const handleAuth = (e: React.FormEvent, _type: "login" | "signup") => {
    e.preventDefault();
    const form = new FormData(e.target as HTMLFormElement);
    setUser({
      name: String(form.get("name")),
      email: String(form.get("email")),
    });
    setPage("map");
  };

  const logout = () => {
    setUser(null);
    setSeats((prev) =>
      prev.map((s) => (s.status === "mine" ? { ...s, status: "free" } : s))
    );
    setPage("login");
  };

  const handleConfirmReservation = (
    updatedSeats: Seat[],
    building: string,
    course: string,
    date: string
  ) => {
    if (!user) return;

    // 写入 storage
    addReservation({
      user,
      seats: updatedSeats.map((s) => s.id),
      building,
      course,
      date,
    });

    // mine -> reserved（立即反映到图上）
    setSeats((prev) =>
      prev.map((s) =>
        updatedSeats.some((u) => u.id === s.id)
          ? { ...s, status: "reserved" }
          : s
      )
    );

    alert(
      `✅ Reservation confirmed for ${updatedSeats
        .map((s) => s.id)
        .join(", ")}\n📍 ${building}\n📘 ${course}\n🕒 ${date}`
    );
    setPage("map");
  };

  // --- 页面切换 ---
  if (page === "login")
    return (
      <AuthCard
        title="Log In to SmartSeat"
        button="Log In"
        switchText="Don't have an account? Sign up"
        onSubmit={(e) => handleAuth(e, "login")}
        onSwitch={() => setPage("signup")}
      />
    );

  if (page === "signup")
    return (
      <AuthCard
        title="Create SmartSeat Account"
        button="Sign Up"
        switchText="Already have an account? Log in"
        onSubmit={(e) => handleAuth(e, "signup")}
        onSwitch={() => setPage("login")}
      />
    );

  if (page === "reserve")
    return (
      <ReservePage
        user={user!}
        seats={selected}
        onBack={() => setPage("map")}
        onConfirm={handleConfirmReservation}
      />
    );

  return (
    <div className="flex min-h-screen flex-col bg-slate-50 animate-fadeIn">
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <header className="flex justify-between items-center py-6">
          <h1 className="text-2xl font-bold text-indigo-700">SmartSeat</h1>
          <div className="text-sm text-slate-600">
            Hello, <b>{user?.name}</b>
            <button
              onClick={logout}
              className="text-indigo-600 hover:underline ml-2"
            >
              Logout
            </button>
          </div>
        </header>

        <section className="text-center my-8">
          <h2 className="text-3xl font-bold text-slate-800">🪑 Seat Map</h2>
          <p className="text-slate-500 mt-2">
            Click to select or deselect your preferred seat.
          </p>
          <p className="mt-2 text-indigo-600 font-medium">
            {selected.length
              ? `You selected ${selected.length} seat${
                  selected.length > 1 ? "s" : ""
                }.`
              : "No seats selected yet."}
          </p>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
            {seats.map((s) => (
              <SeatButton key={s.id} seat={s} onPick={toggleSeat} />
            ))}
          </div>

          {selected.length > 0 && (
            <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button
                onClick={() =>
                  setSeats((prev) =>
                    prev.map((s) =>
                      s.status === "mine" ? { ...s, status: "free" } : s
                    )
                  )
                }
                className="rounded-lg border border-slate-300 px-4 py-2 text-sm text-slate-700 hover:bg-slate-100"
              >
                Clear Selection
              </button>
              <div className="flex justify-end">
                <button
                  onClick={() => setPage("reserve")}
                  className="rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white shadow-md hover:bg-indigo-700"
                >
                  Reserve Seat
                </button>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

// ---- UI 子组件 ----
const AuthCard = ({
  title,
  button,
  switchText,
  onSubmit,
  onSwitch,
}: {
  title: string;
  button: string;
  switchText: string;
  onSubmit: (e: React.FormEvent) => void;
  onSwitch: () => void;
}) => (
  <div className="flex items-center justify-center min-h-screen bg-slate-50">
    <form
      onSubmit={onSubmit}
      className="bg-white p-8 rounded-xl shadow-md w-full max-w-sm space-y-5"
    >
      <h1 className="text-2xl font-bold text-center text-indigo-700">
        {title}
      </h1>
      <input
        name="name"
        required
        placeholder="Your Name"
        className="w-full border border-slate-300 rounded-md px-3 py-2"
      />
      <input
        name="email"
        required
        type="email"
        placeholder="Email"
        className="w-full border border-slate-300 rounded-md px-3 py-2"
      />
      <input
        name="password"
        required
        type="password"
        placeholder="Password"
        className="w-full border border-slate-300 rounded-md px-3 py-2"
      />
      <button
        type="submit"
        className="w-full bg-indigo-600 text-white rounded-md py-2 font-semibold hover:bg-indigo-700"
      >
        {button}
      </button>
      <p
        onClick={onSwitch}
        className="text-center text-sm text-indigo-600 hover:underline cursor-pointer"
      >
        {switchText}
      </p>
    </form>
  </div>
);

const ReservePage = ({
  user,
  seats,
  onBack,
  onConfirm,
}: {
  user: User;
  seats: Seat[];
  onBack: () => void;
  onConfirm: (
    updatedSeats: Seat[],
    building: string,
    course: string,
    date: string
  ) => void;
}) => {
  const [date, setDate] = useState("");
  const [building, setBuilding] = useState("");
  const [course, setCourse] = useState("");

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center px-4">
      <div className="bg-white p-8 rounded-2xl shadow-lg w-full max-w-lg space-y-6 animate-fadeIn">
        <h2 className="text-2xl font-bold text-center text-indigo-700">
          Confirm Your Reservation
        </h2>
        <p className="text-center text-slate-600">
          {user.name} ({user.email})
        </p>
        <p className="text-center text-slate-700">
          Seats selected:{" "}
          <b className="text-indigo-700">{seats.map((s) => s.id).join(", ")}</b>
        </p>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2">
            <Calendar className="w-4 h-4 text-indigo-600" /> Date & Time
          </label>
          <input
            required
            type="datetime-local"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2">
            <Building2 className="w-4 h-4 text-indigo-600" /> Building
          </label>
          <select
            required
            value={building}
            onChange={(e) => setBuilding(e.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          >
            <option value="">Select a building</option>
            <option>Main Campus</option>
            <option>Library Block</option>
            <option>Innovation Hub</option>
            <option>Engineering Tower</option>
            <option>Business Centre</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-indigo-600" /> Course Type
          </label>
          <select
            required
            value={course}
            onChange={(e) => setCourse(e.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-slate-900 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          >
            <option value="">Select course type</option>
            <option>Lecture</option>
            <option>Tutorial</option>
            <option>Workshop</option>
            <option>Lab Session</option>
            <option>Group Discussion</option>
          </select>
        </div>

        <div className="flex justify-between mt-6">
          <button
            onClick={onBack}
            className="px-4 py-2 rounded-md border border-slate-300 text-slate-700 hover:bg-slate-100"
          >
            ← Back
          </button>
          <button
            onClick={() => onConfirm(seats, building, course, date)}
            className="px-6 py-2 rounded-md bg-indigo-600 text-white font-semibold hover:bg-indigo-700"
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
};

const SeatButton = ({
  seat,
  onPick,
}: {
  seat: Seat;
  onPick: (s: Seat) => void;
}) => {
  const style =
    seat.status === "free"
      ? "bg-emerald-50 text-emerald-700 ring-emerald-200 hover:ring-2"
      : seat.status === "reserved"
      ? "bg-amber-50 text-amber-700 cursor-not-allowed"
      : "bg-indigo-600 text-white ring-indigo-300 hover:ring-2";
  const clickable = seat.status !== "reserved";
  return (
    <button
      disabled={!clickable}
      onClick={() => clickable && onPick(seat)}
      className={`h-12 rounded-md text-sm font-medium transition ring-1 ring-transparent ${style}`}
    >
      {seat.id}
    </button>
  );
};

export default SeatMap;
