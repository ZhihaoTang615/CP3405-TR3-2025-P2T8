// src/hooks/useSeats.ts
import type { Seat } from "@/types";

// 稳定伪随机（按房间键种子）
function seedFromString(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function seededRandom(seed: number) {
  let x = seed || 123456789;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return (x >>> 0) / 4294967295;
  };
}

function buildSquareSeats(roomKey: string): Seat[] {
  const rng = seededRandom(seedFromString(`${roomKey}-square`));
  const out: Seat[] = [];
  for (let t = 1; t <= 6; t++) {
    for (let s = 1; s <= 6; s++) {
      const id = `T${t}-S${s}`;
      out.push({
        id,
        label: id,
        status: rng() < 0.12 ? "reserved" : "free",
      });
    }
  }
  return out;
}

function buildRoundSeats(roomKey: string): Seat[] {
  const rng = seededRandom(seedFromString(`${roomKey}-round`));
  const out: Seat[] = [];
  for (let t = 1; t <= 6; t++) {
    for (let s = 1; s <= 6; s++) {
      const id = `R${t}-S${s}`;
      out.push({
        id,
        label: id,
        status: rng() < 0.12 ? "reserved" : "free",
      });
    }
  }
  return out;
}

function buildLectureSeats(roomKey: string): Seat[] {
  const rng = seededRandom(seedFromString(`${roomKey}-lecture`));
  const out: Seat[] = [];
  const total = 200;
  for (let i = 1; i <= total; i++) {
    const id = `L${String(i).padStart(3, "0")}`;
    out.push({
      id,
      label: id,
      status: rng() < 0.06 ? "reserved" : "free",
    });
  }
  return out;
}

export function useSeats(
  layout: "square" | "round" | "lecture" | "none",
  roomKey: string
): Seat[] {
  if (!roomKey || !layout || layout === "none") return [];
  switch (layout) {
    case "square":
      return buildSquareSeats(roomKey);
    case "round":
      return buildRoundSeats(roomKey);
    case "lecture":
      return buildLectureSeats(roomKey);
    default:
      return [];
  }
}
