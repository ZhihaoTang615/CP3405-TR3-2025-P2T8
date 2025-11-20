// src/pages/BookSeat.tsx
import React, { useMemo, useState } from "react";
import {
  Check,
  MapPin,
  Users,
  Sparkles,
  School,
  ArrowRight,
} from "lucide-react";

// -------------------------------
// Types / constants
// -------------------------------
type CourseType = "Practical" | "Lecture" | "Tutorial" | "Group Discussion";
const COURSE_TYPES: CourseType[] = [
  "Practical",
  "Lecture",
  "Tutorial",
  "Group Discussion",
];

type BlockID = "A" | "B" | "C" | "D" | "E";
const BLOCKS: BlockID[] = ["A", "B", "C", "D", "E"];

const FLOORS: Record<BlockID, number[]> = {
  A: [1, 2, 3],
  B: [1, 2, 3],
  C: [1, 2, 3, 4],
  D: [1],
  E: [1, 2],
};

const ROOMS_PER_FLOOR = 6; // 每层 6 个教室
const ROWS = 5; // 5 排
const COLS = 6; // 6 列（共 30 个座位）

type SeatState = "free" | "held" | "reserved" | "mine";
type Seat = {
  id: string;
  row: number;
  col: number;
  state: SeatState;
};

// 简单窗口朝向规则（为了“窗边优先”规则演示）：A/C/E 左侧有窗；B/D 右侧有窗
const WINDOWS_ON_LEFT: BlockID[] = ["A", "C", "E"];

// -------------------------------
// Helpers
// -------------------------------
function makeRoomCode(block: BlockID, floor: number, roomIdx: number) {
  const idx = String(roomIdx).padStart(2, "0");
  return `${block}-${floor}${idx}`;
}

function generateSeats(block: BlockID, floor: number, room: number): Seat[] {
  // 模拟：不同房间有不同的占用（reserved）模式
  const seed = `${block}${floor}${room}`
    .split("")
    .reduce((a, c) => a + c.charCodeAt(0), 0);
  const rng = mulberry32(seed);

  const arr: Seat[] = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const id = `${makeRoomCode(block, floor, room)}-R${r + 1}C${c + 1}`;
      // ~25% 随机 reserved，避免全空效果
      const reserved = rng() < 0.25;
      arr.push({
        id,
        row: r,
        col: c,
        state: reserved ? "reserved" : "free",
      });
    }
  }
  return arr;
}

// 简易随机数（可复现）
function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 智能推荐：后排优先 → 靠窗优先 → 成组就近（size 可为 1 表示单人）
function recommendSeats(
  seats: Seat[],
  size: number,
  block: BlockID
): Seat[] | null {
  const freeByRow = groupBy(
    seats.filter((s) => s.state === "free"),
    (s) => s.row
  );

  const windowOnLeft = WINDOWS_ON_LEFT.includes(block);

  // 从“后排”开始找连续空位
  for (let r = ROWS - 1; r >= 0; r--) {
    const rowSeats = (freeByRow[r] || []).sort((a, b) => {
      // 窗边优先：靠左则 col 小优先；靠右则 col 大优先
      return windowOnLeft ? a.col - b.col : b.col - a.col;
    });

    // 扫描连续空位
    for (let start = 0; start <= COLS - size; start++) {
      const group = rowSeats.filter(
        (s) => s.col >= start && s.col < start + size
      );
      if (group.length === size) {
        // 连续 size 个空位
        const cols = group.map((g) => g.col).sort((x, y) => x - y);
        // 保障确实连续（避免被排序后跨列）
        let consecutive = true;
        for (let i = 1; i < cols.length; i++) {
          if (cols[i] !== cols[i - 1] + 1) consecutive = false;
        }
        if (consecutive) return group;
      }
    }
  }
  return null;
}

function groupBy<T, K extends string | number>(
  arr: T[],
  key: (x: T) => K
): Record<K, T[]> {
  return arr.reduce((acc, cur) => {
    const k = key(cur);
    (acc[k] ||= []).push(cur);
    return acc;
  }, {} as Record<K, T[]>);
}

// -------------------------------
// UI Components
// -------------------------------
function Pill<T extends string>({
  value,
  current,
  onSelect,
}: {
  value: T;
  current: T;
  onSelect: (v: T) => void;
}) {
  const active = value === current;
  return (
    <button
      onClick={() => onSelect(value)}
      className={`px-3 py-1.5 rounded-full border text-sm ${
        active
          ? "bg-indigo-600 text-white border-indigo-600 shadow-sm"
          : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50"
      }`}
    >
      {value}
    </button>
  );
}

function StepHeader({
  step,
  title,
  icon,
}: {
  step: number;
  title: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <div className="size-6 rounded-full bg-indigo-600 text-white text-xs grid place-items-center">
        {step}
      </div>
      <div className="flex items-center gap-2 text-slate-800 font-medium">
        {icon}
        <span>{title}</span>
      </div>
    </div>
  );
}

// 单个座位按钮
function SeatDot({
  seat,
  selected,
  onClick,
}: {
  seat: Seat;
  selected: boolean;
  onClick: () => void;
}) {
  const base =
    "w-8 h-8 rounded-md grid place-items-center text-[11px] select-none transition";
  let cls =
    "bg-white border border-slate-300 hover:border-indigo-400 hover:shadow-sm";
  if (seat.state === "reserved")
    cls = "bg-slate-200 border border-slate-300 text-slate-400";
  if (seat.state === "mine") cls = "bg-emerald-600 text-white";
  if (selected) cls = "bg-indigo-600 text-white";

  return (
    <button
      disabled={seat.state === "reserved"}
      onClick={onClick}
      className={`${base} ${cls}`}
      title={seat.id}
      aria-label={seat.id}
    >
      {seat.col + 1}
    </button>
  );
}

// -------------------------------
// Page
// -------------------------------
export default function BookSeat() {
  // Step selections
  const [course, setCourse] = useState<CourseType>("Practical");
  const [block, setBlock] = useState<BlockID>("A");
  const [floor, setFloor] = useState<number>(FLOORS["A"][0]);
  const [room, setRoom] = useState<number>(1);

  // 座位状态
  const [selected, setSelected] = useState<string[]>([]);
  const [partySize, setPartySize] = useState<number>(1);

  const seats = useMemo(
    () => generateSeats(block, floor, room),
    [block, floor, room]
  );

  const roomCode = makeRoomCode(block, floor, room);

  const freeCount = seats.filter((s) => s.state === "free").length;
  const reservedCount = seats.filter((s) => s.state === "reserved").length;

  const handleToggleSeat = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const handleRecommend = () => {
    const group = recommendSeats(seats, Math.max(1, partySize), block);
    if (!group) {
      alert("未找到满足条件的连续空位（可能已满）。");
      return;
    }
    setSelected(group.map((g) => g.id));
  };

  const handleConfirm = () => {
    if (selected.length === 0) {
      alert("请先选择座位");
      return;
    }
    // 这里暂时简单打印，后面可接 Firebase/后端
    console.log("✅ Reservation confirmed:", {
      course,
      block,
      floor,
      room,
      seats: selected,
    });
    alert(`预订成功：${selected.length} 个座位（${roomCode}）`);
  };

  // 切换 Block 时重置 floor/room
  const handleBlock = (b: BlockID) => {
    setBlock(b);
    setFloor(FLOORS[b][0]);
    setRoom(1);
    setSelected([]);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-indigo-700 flex items-center gap-2">
          <Sparkles className="w-6 h-6" /> SmartSeat — Book & Pick Seats
        </h1>
        <p className="text-slate-600 mt-1">
          先选择课程 / 楼栋 / 楼层 / 教室 → 然后在右侧选择座位（支持智能推荐）。
        </p>
      </header>

      {/* Step 1: 课程类型 */}
      <section className="mb-4">
        <StepHeader
          step={1}
          title="选择课程类型"
          icon={<School className="w-4 h-4" />}
        />
        <div className="flex flex-wrap gap-2">
          {COURSE_TYPES.map((t) => (
            <Pill key={t} value={t} current={course} onSelect={setCourse} />
          ))}
        </div>
      </section>

      {/* Step 2: Block */}
      <section className="mb-4">
        <StepHeader
          step={2}
          title="选择教学楼 Block"
          icon={<MapPin className="w-4 h-4" />}
        />
        <div className="flex flex-wrap gap-2">
          {BLOCKS.map((b) => (
            <Pill key={b} value={b} current={block} onSelect={handleBlock} />
          ))}
        </div>
      </section>

      {/* Step 3 + 4: Floor + Room */}
      <section className="mb-6">
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <StepHeader
              step={3}
              title="选择楼层"
              icon={<ArrowRight className="w-4 h-4" />}
            />
            <div className="flex flex-wrap gap-2">
              {FLOORS[block].map((f) => (
                <Pill
                  key={f}
                  value={f as any}
                  current={floor as any}
                  onSelect={setFloor as any}
                />
              ))}
            </div>
          </div>
          <div>
            <StepHeader
              step={4}
              title="选择教室"
              icon={<ArrowRight className="w-4 h-4" />}
            />
            <div className="flex flex-wrap gap-2">
              {Array.from({ length: ROOMS_PER_FLOOR }, (_, i) => i + 1).map(
                (r) => (
                  <Pill
                    key={r}
                    value={r as any}
                    current={room as any}
                    onSelect={setRoom as any}
                  />
                )
              )}
            </div>
          </div>
        </div>
        <p className="text-sm text-slate-500 mt-2">
          当前：<span className="font-medium">{roomCode}</span> ・空位{" "}
          {freeCount} ・已占 {reservedCount}
          {WINDOWS_ON_LEFT.includes(block) ? " ・窗户在左侧" : " ・窗户在右侧"}
        </p>
      </section>

      {/* Seat map + side panel */}
      <div className="grid lg:grid-cols-[1fr_320px] gap-6">
        {/* Seat grid */}
        <div className="rounded-xl border bg-white p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="text-slate-700 font-medium">
              座位图（{ROWS} × {COLS}）
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1 text-xs text-slate-500">
                <span className="w-3 h-3 rounded-sm border border-slate-300 bg-white inline-block" />{" "}
                空位
              </div>
              <div className="flex items-center gap-1 text-xs text-slate-500">
                <span className="w-3 h-3 rounded-sm bg-slate-300 inline-block" />{" "}
                已占
              </div>
              <div className="flex items-center gap-1 text-xs text-slate-500">
                <span className="w-3 h-3 rounded-sm bg-indigo-600 inline-block" />{" "}
                已选
              </div>
            </div>
          </div>

          <div
            className="grid gap-2"
            style={{ gridTemplateRows: `repeat(${ROWS}, minmax(0,1fr))` }}
          >
            {Array.from({ length: ROWS }).map((_, r) => (
              <div
                key={r}
                className="grid gap-2"
                style={{
                  gridTemplateColumns: `repeat(${COLS}, minmax(0,1fr))`,
                }}
              >
                {seats
                  .filter((s) => s.row === r)
                  .map((s) => (
                    <SeatDot
                      key={s.id}
                      seat={s}
                      selected={selected.includes(s.id)}
                      onClick={() => handleToggleSeat(s.id)}
                    />
                  ))}
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <aside className="rounded-xl border bg-white p-4 h-max">
          <div className="flex items-center justify-between">
            <div className="font-semibold text-slate-800">选择与推荐</div>
            <Users className="w-4 h-4 text-slate-500" />
          </div>

          <div className="mt-3">
            <label className="block text-sm text-slate-600 mb-1">
              同行人数（智能推荐）
            </label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={1}
                max={6}
                value={partySize}
                onChange={(e) =>
                  setPartySize(
                    Math.max(1, Math.min(6, Number(e.target.value) || 1))
                  )
                }
                className="w-24 rounded-md border border-slate-300 px-2 py-1 text-sm"
              />
              <button
                onClick={handleRecommend}
                className="inline-flex items-center gap-2 rounded-md bg-indigo-600 text-white px-3 py-1.5"
              >
                <Sparkles className="w-4 h-4" /> 智能推荐
              </button>
            </div>
          </div>

          <div className="mt-4">
            <div className="text-sm text-slate-600 mb-1">已选座位</div>
            {selected.length === 0 ? (
              <div className="text-sm text-slate-400">尚未选择</div>
            ) : (
              <ul className="text-sm text-slate-700 space-y-1">
                {selected.map((id) => (
                  <li key={id} className="flex items-center gap-2">
                    <Check className="w-4 h-4 text-emerald-600" />
                    {id}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <button
            onClick={handleConfirm}
            className="mt-5 w-full rounded-md bg-emerald-600 text-white py-2 font-medium hover:bg-emerald-700"
          >
            确认预订
          </button>
        </aside>
      </div>
    </div>
  );
}
