// src/pages/LecturerReserve.tsx
import React, { useMemo, useState } from "react";
import {
  Building2,
  BookOpen,
  Calendar,
  Map,
  MapPin,
  Info,
  CheckCircle2,
  Clock,
} from "lucide-react";
import roomsData from "@/data/rooms.json";
import { useAuth } from "@/lib/auth";
import { addReservation } from "@/lib/reservation";

type CourseType =
  | "Lecture"
  | "Tutorial"
  | "Practical"
  | "Lab"
  | "Group Discussion";

type BlockKey = keyof typeof roomsData;

export default function LecturerReserve() {
  const { user } = useAuth();
  const uid = user?.uid;

  const [courseType, setCourseType] = useState<CourseType | "">("");
  const [block, setBlock] = useState<BlockKey | "">("");
  const [floor, setFloor] = useState<string>("");
  const [room, setRoom] = useState<string>("");
  const [dateTime, setDateTime] = useState("");
  const [duration, setDuration] = useState<string>("2");
  const [isDateConfirmed, setIsDateConfirmed] = useState(false);
  const [note, setNote] = useState("");

  const floors = useMemo(
    () =>
      block ? Object.keys(roomsData[block].floors).sort((a, b) => +a - +b) : [],
    [block]
  );

  const availableRooms = useMemo(
    () => (block && floor ? roomsData[block].floors[floor]?.rooms ?? [] : []),
    [block, floor]
  );

  const canConfirm =
    !!uid &&
    !!courseType &&
    !!block &&
    !!floor &&
    !!room &&
    !!dateTime &&
    isDateConfirmed;

  const handleConfirm = async () => {
    if (!uid) {
      alert("⚠️ Please login again as a lecturer.");
      return;
    }
    if (!canConfirm) return;

    const building = `${room} • Block ${block} • Floor ${floor}`;
    await addReservation(uid, {
      building,
      course: String(courseType),
      seats: [],
      date: dateTime,
      name: user?.email || "Lecturer",
      email: user?.email || "",
      durationHours: Number(duration),
      fromRole: "lecturer",
      note: note || "",
    });

    alert("✅ Room reservation created successfully!");
    setRoom("");
    setIsDateConfirmed(false);
    setNote("");
  };

  const Step = ({
    ok,
    label,
    n,
  }: {
    ok: boolean;
    label: string;
    n: number;
  }) => (
    <div className="flex items-center gap-2 text-sm">
      <span
        className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-semibold ${
          ok
            ? "bg-emerald-500/30 text-emerald-600 dark:text-emerald-300 ring-1 ring-emerald-400"
            : "bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-400"
        }`}
      >
        {ok ? <CheckCircle2 className="h-3 w-3" /> : n}
      </span>
      <span
        className={
          ok
            ? "text-emerald-600 dark:text-emerald-300"
            : "text-slate-600 dark:text-slate-400"
        }
      >
        {label}
      </span>
    </div>
  );

  return (
    <div className="mx-auto max-w-6xl p-6 route-fade text-slate-900 dark:text-white">
      {/* ===== HEADER ===== */}
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-blue-600 dark:text-blue-300 drop-shadow-sm">
          🎓 Lecturer Room Reservation
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-slate-600 dark:text-slate-400">
          Plan your teaching sessions by reserving a full classroom. This page
          is dedicated to lecturers only.
        </p>
      </header>

      {/* ===== Step instruction ===== */}
      <div className="card glass mb-8">
        <div className="flex items-start gap-3">
          <div className="rounded-md bg-blue-500/20 p-2">
            <Info className="h-5 w-5 text-blue-500 dark:text-blue-300" />
          </div>
          <div className="text-sm">
            <p className="font-medium text-blue-600 dark:text-blue-300">
              Follow these 4 steps to reserve a room
            </p>
            <ul className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <Step ok={!!courseType} n={1} label="Choose Class Type" />
              <Step
                ok={!!block && !!floor && !!room}
                n={2}
                label="Select Room"
              />
              <Step
                ok={!!dateTime && isDateConfirmed}
                n={3}
                label="Confirm Date & Time"
              />
              <Step ok={canConfirm} n={4} label="Review & Confirm" />
            </ul>
          </div>
        </div>
      </div>

      {/* ===== Form area ===== */}
      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <SelectCard
          icon={<BookOpen />}
          title="Class Type"
          value={courseType}
          onChange={(v) => setCourseType(v as CourseType)}
          options={[
            "Lecture",
            "Tutorial",
            "Practical",
            "Lab",
            "Group Discussion",
          ]}
        />

        <SelectCard
          icon={<Building2 />}
          title="Block"
          value={block}
          onChange={(v) => {
            setBlock(v as BlockKey);
            setFloor("");
            setRoom("");
          }}
          options={Object.keys(roomsData)}
        />

        <SelectCard
          icon={<MapPin />}
          title="Floor"
          value={floor}
          onChange={(v) => {
            setFloor(String(v));
            setRoom("");
          }}
          options={floors}
          disabled={!block}
        />

        <SelectCard
          icon={<Map />}
          title="Classroom"
          value={room}
          onChange={(v) => setRoom(String(v))}
          options={availableRooms}
          disabled={!block || !floor}
        />

        {/* ===== Time & Duration ===== */}
        <div className="card glass">
          <label className="flex items-center gap-2 text-sm mb-1 font-semibold text-blue-600 dark:text-blue-300">
            <Calendar className="w-4 h-4" /> Class Start Time
          </label>
          <input
            type="datetime-local"
            value={dateTime}
            onChange={(e) => {
              setDateTime(e.target.value);
              setIsDateConfirmed(false);
            }}
            className="input"
          />

          <label className="flex items-center gap-2 text-sm mb-1 mt-4 font-semibold text-blue-600 dark:text-blue-300">
            <Clock className="w-4 h-4" /> Duration (hours)
          </label>
          <input
            type="number"
            min={1}
            max={6}
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
            className="input"
          />

          {!isDateConfirmed ? (
            <button
              onClick={() => setIsDateConfirmed(true)}
              disabled={!dateTime}
              className="btn-primary mt-3 w-full disabled:opacity-50"
            >
              Confirm Date & Time
            </button>
          ) : (
            <p className="text-emerald-600 dark:text-emerald-300 text-sm mt-2 flex items-center gap-1">
              <CheckCircle2 className="h-4 w-4" /> Date & Time Confirmed
            </p>
          )}
        </div>

        {/* ===== Notes ===== */}
        <div className="card glass">
          <label className="block text-sm font-semibold text-blue-600 dark:text-blue-300 mb-1">
            Notes (optional)
          </label>
          <textarea
            placeholder="e.g. mid-term review session"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            className="input min-h-[96px] resize-none"
          />
        </div>
      </section>

      {/* ===== Summary ===== */}
      <section className="mt-10 card glass">
        {!block || !floor || !room ? (
          <p className="py-6 text-center text-slate-600 dark:text-slate-400">
            📍 Select Block, Floor & Classroom to review your booking.
          </p>
        ) : !dateTime || !isDateConfirmed ? (
          <p className="py-6 text-center text-slate-600 dark:text-slate-400">
            📅 Confirm date & time to proceed.
          </p>
        ) : (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-blue-600 dark:text-blue-300">
              Booking Summary
            </h2>
            <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
              <li>
                <span className="text-slate-500 dark:text-slate-400">
                  Class type:
                </span>{" "}
                {courseType}
              </li>
              <li>
                <span className="text-slate-500 dark:text-slate-400">
                  Room:
                </span>{" "}
                {room} • Block {block} • Floor {floor}
              </li>
              <li>
                <span className="text-slate-500 dark:text-slate-400">
                  Start:
                </span>{" "}
                {new Date(dateTime).toLocaleString()}
              </li>
              <li>
                <span className="text-slate-500 dark:text-slate-400">
                  Duration:
                </span>{" "}
                {duration} hours
              </li>
              <li>
                <span className="text-slate-500 dark:text-slate-400">
                  Lecturer:
                </span>{" "}
                {user?.email}
              </li>
            </ul>
            <button
              onClick={handleConfirm}
              disabled={!canConfirm}
              className="btn-primary px-6 py-2.5 disabled:opacity-50"
            >
              Confirm Room Reservation
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

function SelectCard({
  icon,
  title,
  value,
  onChange,
  options,
  disabled = false,
}: any) {
  return (
    <div className="card glass">
      <label className="flex items-center gap-2 text-sm mb-1 font-semibold text-blue-600 dark:text-blue-300">
        {icon} {title}
      </label>
      <select
        disabled={disabled}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="input disabled:opacity-50"
      >
        <option value="">Select {title.toLowerCase()}</option>
        {options.map((opt: string, i: number) => (
          <option key={i} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}
