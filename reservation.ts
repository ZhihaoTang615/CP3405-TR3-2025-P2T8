// src/lib/reservation.ts
import {
  addDoc,
  collection,
  doc,
  getDocs,
  orderBy,
  query,
  serverTimestamp,
  where,
  writeBatch,
} from "firebase/firestore";
import { db } from "@/firebase";

export type Reservation = {
  id?: string;
  uid: string;
  building: string; // "B2-03 • Block B • Floor 2"
  course: string; // "Lecture" | "Tutorial" | ...
  seats: string[]; // ["T6-S5", "T6-S6"]
  date: string; // ISO or "YYYY-MM-DDTHH:mm"
  name?: string; // 可选：如果你要保存
  email?: string; // 可选：如果你要保存
  createdAt?: any; // Firestore Timestamp
};

/** 新增一条预约 */
export async function addReservation(
  uid: string,
  data: Omit<Reservation, "uid">
) {
  const col = collection(db, "reservations");
  const payload: Reservation = {
    uid,
    ...data,
    createdAt: serverTimestamp(),
  };
  const ref = await addDoc(col, payload);
  return ref.id;
}

/** 查询当前用户的所有预约（按时间倒序） */
export async function listReservations(uid: string): Promise<Reservation[]> {
  const col = collection(db, "reservations");
  // 有的旧文档可能没有 createdAt，排序时可能报错；我们先 where，再取回后本地按时间降序兜底
  const q = query(col, where("uid", "==", uid), orderBy("createdAt", "desc"));
  try {
    const snap = await getDocs(q);
    return snap.docs.map((d) => ({ id: d.id, ...(d.data() as any) }));
  } catch {
    // 如果因为部分文档缺少 createdAt 导致排序报错，用无序查询 + 本地排序兜底
    const snap = await getDocs(query(col, where("uid", "==", uid)));
    return snap.docs
      .map((d) => ({ id: d.id, ...(d.data() as any) }))
      .sort((a: any, b: any) => {
        const ta = a?.createdAt?.toMillis?.() ?? 0;
        const tb = b?.createdAt?.toMillis?.() ?? 0;
        return tb - ta;
      });
  }
}

/** 清空当前用户的预约 */
export async function clearReservations(uid: string) {
  const col = collection(db, "reservations");
  const q = query(col, where("uid", "==", uid));
  const snap = await getDocs(q);

  if (snap.empty) return;

  const batch = writeBatch(db);
  snap.docs.forEach((d) => batch.delete(doc(db, "reservations", d.id)));
  await batch.commit();
}
