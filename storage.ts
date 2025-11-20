// ✅ Firestore version
import { db } from "@/firebase";
import {
  collection,
  query,
  where,
  getDocs,
  orderBy,
  deleteDoc,
} from "firebase/firestore";

export async function listMyReservations(uid: string) {
  const q = query(
    collection(db, "reservations"),
    where("uid", "==", uid),
    orderBy("createdAt", "desc")
  );

  const snap = await getDocs(q);
  return snap.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
}

export async function clearMyReservations(uid: string) {
  const q = query(collection(db, "reservations"), where("uid", "==", uid));
  const snap = await getDocs(q);

  for (const docSnap of snap.docs) {
    await deleteDoc(docSnap.ref);
  }
}
