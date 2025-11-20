// src/hooks/useSeatsFirebase.ts
import { useEffect, useMemo, useState } from "react";
import { db } from "@/lib/firebase";
import {
  collection,
  doc,
  getDoc,
  getDocs,
  onSnapshot,
  runTransaction,
  setDoc,
} from "firebase/firestore";
import type { Seat } from "@/types";

// 生成器与你本地 useSeats 保持一致
function genIds(
  layout: "square" | "round" | "lecture",
  roomKey: string
): { id: string; label: string }[] {
  const out: { id: string; label: string }[] = [];
  if (layout === "lecture") {
    for (let i = 1; i <= 200; i++) {
      const id = `L${String(i).padStart(3, "0")}`;
      out.push({ id, label: id });
    }
  } else {
    const prefix = layout === "square" ? "T" : "R";
    for (let t = 1; t <= 6; t++) {
      for (let s = 1; s <= 6; s++) {
        const id = `${prefix}${t}-S${s}`;
        out.push({ id, label: id });
      }
    }
  }
  return out;
}

/** 首次进入房间时，如果 seats 不存在则初始化；随后订阅变更 */
export function useSeatsFirebase(
  layout: "square" | "round" | "lecture" | "none",
  roomKey: string,
  uid?: string
) {
  const [seats, setSeats] = useState<Seat[]>([]);
  const valid = layout !== "none" && roomKey && uid;

  useEffect(() => {
    if (!valid) {
      setSeats([]);
      return;
    }

    const roomRef = doc(db, "rooms", roomKey);
    const seatsCol = collection(roomRef, "seats");

    (async () => {
      const roomSnap = await getDoc(roomRef);
      if (!roomSnap.exists()) {
        await setDoc(roomRef, {
          layout,
          seats: layout === "lecture" ? 200 : 36,
        });
        // 初始化子集合 seats
        const ids = genIds(layout as any, roomKey);
        await Promise.all(
          ids.map(({ id, label }) =>
            setDoc(doc(seatsCol, id), {
              status: "free",
              reservedBy: null,
              label,
            })
          )
        );
      }
      // 订阅
      return onSnapshot(seatsCol, (snap) => {
        const list: Seat[] = [];
        snap.forEach((d) => {
          const data = d.data() as any;
          list.push({
            id: d.id,
            label: data.label,
            status:
              data.status === "free"
                ? "free"
                : data.reservedBy === uid
                ? "mine"
                : "reserved",
          });
        });
        // 保持原顺序
        setSeats(
          list.sort((a, b) => (a.label || a.id).localeCompare(b.label || b.id))
        );
      });
    })();
  }, [valid, roomKey, layout, uid]);

  const toggle = async (seat: Seat) => {
    if (!valid) return;
    const seatRef = doc(db, "rooms", roomKey, "seats", seat.id);
    await runTransaction(db, async (tx) => {
      const s = await tx.get(seatRef);
      const data = s.data() as any;
      if (!data) throw new Error("seat missing");
      const isMine = data.reservedBy === uid;
      if (data.status === "free") {
        tx.set(
          seatRef,
          { ...data, status: "reserved", reservedBy: uid },
          { merge: true }
        );
      } else if (isMine) {
        tx.set(
          seatRef,
          { ...data, status: "free", reservedBy: null },
          { merge: true }
        );
      } else {
        throw new Error("Seat taken");
      }
    });
  };

  return { seats, toggle };
}
