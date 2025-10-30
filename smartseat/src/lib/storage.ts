// src/lib/storage.ts
import type { Reservation } from "@/types";

const STORAGE_KEY = "smartseat.reservations";

export function getReservations(): Reservation[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

export function saveReservations(list: Reservation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

export function addReservation(
  r: Omit<Reservation, "id" | "createdAt">
): Reservation {
  const all = getReservations();
  const newItem: Reservation = {
    ...r,
    id: Date.now(),
    createdAt: new Date().toISOString(),
  };
  all.push(newItem);
  saveReservations(all);
  return newItem;
}

export function deleteReservation(id: number) {
  const next = getReservations().filter((x) => x.id !== id);
  saveReservations(next);
}

export function getReservedSeatIds(): string[] {
  // 汇总所有未过期/已过期都视为占用（纯前端 demo）
  const all = getReservations();
  // 若需要“过期释放”，可在此根据日期过滤
  const ids = new Set<string>();
  all.forEach((r) => r.seats.forEach((s) => ids.add(s)));
  return Array.from(ids);
}
