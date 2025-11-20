import React, { useMemo, useState, useEffect } from "react";
import {
  Building2,
  BookOpen,
  Calendar,
  Map,
  MapPin,
  Info,
  CheckCircle2,
} from "lucide-react";
import type { Seat } from "@/types";
import { useAuth } from "@/lib/auth";
import { addReservation } from "@/lib/reservation";
import roomsData from "@/data/rooms.json";
import { useSeats } from "@/hooks/useSeats";

type CourseType = "Lecture" | "Tutorial" | "Practical" | "Group Discussion";
type BlockKey = keyof typeof roomsData;
type Layout = "square" | "round" | "lecture" | "none";

function buildRoomKey(block: string, floor: string | number, room: string) {
  return [block || "_", floor || "_", room || "_"].join("-");
}

export default function Reserve() {
  const { user } = useAuth();
  const uid = user?.uid;
  const [courseType, setCourseType] = useState<CourseType | "">("");
  const [block, setBlock] = useState<BlockKey | "">("");
  const [floor, setFloor] = useState<string>("");
  const [room, setRoom] = useState<string>("");
  const [dateTime, setDateTime] = useState("");
  const [isDateConfirmed, setIsDateConfirmed] = useState(false);
  const [name, setName] = useState(user?.displayName || "");
  const [email, setEmail] = useState(user?.email || "");

  const floors = useMemo(
    () =>
      block ? Object.keys(roomsData[block].floors).sort((a, b) => +a - +b) : [],
    [block]
  );
  const availableRooms = useMemo(
    () => (block && floor ? roomsData[block].floors[floor]?.rooms ?? [] : []),
    [block, floor]
  );
  const layout: Layout = useMemo(
    () =>
      block && floor
        ? roomsData[block].floors[floor]?.layout ?? "none"
        : "none",
    [block, floor]
  );

  const roomKey = buildRoomKey(String(block), floor, room);
  const [seats, setSeats] = useState<Seat[]>([]);
  const generated = useSeats(layout, roomKey);
  useEffect(() => setSeats(generated), [roomKey, layout, generated.length]);
  const selected = seats.filter((s) => s.status === "mine");

  const toggleSeat = (seat: Seat) =>
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

  const clearSelection = () =>
    setSeats((prev) =>
      prev.map((s) => (s.status === "mine" ? { ...s, status: "free" } : s))
    );

  const canConfirm =
    !!courseType &&
    !!block &&
    !!floor &&
    !!room &&
    !!dateTime &&
    !!name &&
    !!email &&
    selected.length > 0;

  const confirmReservation = async () => {
    if (!uid) return alert("⚠️ Please login again");
    if (!canConfirm) return;
    const building = `${room} • Block ${block} • Floor ${floor}`;

    await addReservation(uid, {
      building,
      course: String(courseType),
      seats: selected.map((s) => s.id),
      date: dateTime,
      name,
      email,
    });

    setSeats((prev) =>
      prev.map((s) =>
        selected.some((x) => x.id === s.id) ? { ...s, status: "reserved" } : s
      )
    );
    alert("✅ Reservation successful!");
    clearSelection();
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
            ? "bg-emerald-500/30 text-emerald-300 ring-1 ring-emerald-400"
            : "bg-slate-700/40 text-slate-400"
        }`}
      >
        {ok ? <CheckCircle2 className="h-3 w-3" /> : n}
      </span>
      <span className={`${ok ? "text-emerald-300" : "text-slate-400"}`}>
        {label}
      </span>
    </div>
  );

  return (
    <div className="mx-auto max-w-6xl p-6 route-fade">
      <header className="text-center mb-8">
        <h1 className="text-3xl font-bold text-blue-300 drop-shadow-sm">
          🎯 Reserve a Seat
        </h1>
        <p className="text-muted mt-2">Pick course → location → date → seats</p>
      </header>

      <div className="card glass mb-8">
        <div className="flex items-start gap-3">
          <div className="rounded-md bg-blue-500/20 p-2">
            <Info className="h-5 w-5 text-blue-400" />
          </div>
          <div className="text-sm">
            <p className="font-medium text-blue-200">
              Tips: follow the 4 steps below
            </p>
            <ul className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <Step ok={!!courseType} n={1} label="Choose Course" />
              <Step
                ok={!!block && !!floor && !!room}
                n={2}
                label="Select Location"
              />
              <Step
                ok={!!dateTime && isDateConfirmed}
                n={3}
                label="Confirm Date & Time"
              />
              <Step
                ok={selected.length > 0}
                n={4}
                label="Pick Seats & Confirm"
              />
            </ul>
          </div>
        </div>
      </div>

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <SelectCard
          icon={<BookOpen />}
          title="Course Type"
          value={courseType}
          onChange={(v) => setCourseType(v as CourseType)}
          options={["Lecture", "Tutorial", "Practical", "Group Discussion"]}
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

        <div className="card glass">
          <label className="flex items-center gap-2 text-sm mb-1 font-medium text-blue-300">
            <Calendar className="w-4 h-4" /> Select Date & Time
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
          {!isDateConfirmed ? (
            <button
              onClick={() => setIsDateConfirmed(true)}
              disabled={!dateTime}
              className="btn-primary mt-3 w-full disabled:opacity-50"
            >
              Confirm Date & Time
            </button>
          ) : (
            <p className="text-emerald-400 text-sm mt-2 flex items-center gap-1">
              <CheckCircle2 className="h-4 w-4" /> Date confirmed
            </p>
          )}
        </div>

        <div className="card glass">
          <label className="block text-sm font-medium text-blue-300 mb-1">
            Your Info
          </label>
          <input
            placeholder="Full name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="input mb-2"
          />
          <input
            placeholder="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="input"
          />
        </div>
      </section>

      <section className="mt-10 card glass">
        {!isDateConfirmed ? (
          <p className="text-muted text-center py-6">
            📅 Confirm date & time to view seats.
          </p>
        ) : !block || !floor || !room ? (
          <p className="text-muted">Pick Block, Floor and Classroom.</p>
        ) : seats.length === 0 ? (
          <p className="text-muted">No seats for this room.</p>
        ) : (
          <>
            <div className="mb-3 flex items-center gap-4 text-xs text-slate-400">
              <Legend dot="seat-free" label="Free" />
              <Legend dot="seat-mine" label="Yours" />
              <Legend dot="seat-reserved" label="Reserved" />
            </div>

            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns:
                  layout === "lecture" ? "repeat(20,1fr)" : "repeat(6,1fr)",
              }}
            >
              {seats.map((s) => (
                <SeatButton key={s.id} seat={s} onPick={toggleSeat} />
              ))}
            </div>

            <div className="mt-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div className="text-sm text-slate-400">
                {selected.length ? (
                  <>
                    Selected:{" "}
                    <b className="text-blue-300">
                      {selected.map((s) => s.label || s.id).join(", ")}
                    </b>
                  </>
                ) : (
                  "No seats selected."
                )}
              </div>
              <div className="flex gap-3">
                <button
                  onClick={clearSelection}
                  className="btn-ghost px-4 py-2"
                >
                  Clear
                </button>
                <button
                  onClick={confirmReservation}
                  disabled={!canConfirm}
                  className="btn-primary px-6 py-2.5 disabled:opacity-60"
                >
                  Confirm Reservation
                </button>
              </div>
            </div>
          </>
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
}) {
  return (
    <div className="card glass">
      <label className="flex items-center gap-2 text-sm mb-1 font-medium text-blue-300">
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

function SeatButton({
  seat,
  onPick,
}: {
  seat: Seat;
  onPick: (s: Seat) => void;
}) {
  const cls =
    seat.status === "free"
      ? "seat-free"
      : seat.status === "reserved"
      ? "seat-reserved cursor-not-allowed"
      : "seat-mine";
  return (
    <button
      disabled={seat.status === "reserved"}
      onClick={() => onPick(seat)}
      className={`h-10 rounded-md text-xs font-medium text-center transition ${cls}`}
      title={seat.label || seat.id}
    >
      {seat.label || seat.id}
    </button>
  );
}

function Legend({ dot, label }: { dot: string; label: string }) {
  return (
    <span className="flex items-center gap-2 text-slate-400">
      <span className={`h-3 w-3 rounded ${dot}`} />
      <span>{label}</span>
    </span>
  );
}
