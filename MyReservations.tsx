import React, { useEffect, useState } from "react";
import {
  collection,
  query,
  where,
  onSnapshot,
  deleteDoc,
  getDocs,
  doc,
} from "firebase/firestore";
import { db } from "@/firebase";
import { useAuth } from "@/lib/auth";
import { Trash2, Calendar } from "lucide-react";

export default function MyReservations() {
  const { user } = useAuth();
  const uid = user?.uid;
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!uid) return;
    const q = query(collection(db, "reservations"), where("uid", "==", uid));
    const unsub = onSnapshot(q, (snap) => {
      const data = snap.docs.map((d) => ({ id: d.id, ...d.data() }));
      setItems(data);
      setLoading(false);
    });
    return () => unsub();
  }, [uid]);

  const clearAll = async () => {
    if (!uid) return;
    if (!confirm("Delete all your reservations?")) return;
    const q = query(collection(db, "reservations"), where("uid", "==", uid));
    const docsSnap = await getDocs(q);
    for (const d of docsSnap.docs)
      await deleteDoc(doc(db, "reservations", d.id));
  };

  return (
    <div className="mx-auto max-w-4xl p-6 route-fade">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-3xl font-bold text-blue-300">
          {" "}
          {/* THEME FIX */}
          My Reservations
        </h1>
        <button
          onClick={clearAll}
          className="btn-ghost glass rounded-md px-3 py-2 flex items-center gap-2 hover:bg-blue-500/20"
        >
          <Trash2 className="h-4 w-4" />
          Clear All
        </button>
      </div>

      {loading ? (
        <p className="text-slate-400">Loading…</p>
      ) : items.length === 0 ? (
        <div className="glass card text-center text-slate-400">
          No reservations yet.
        </div>
      ) : (
        <ul className="space-y-3">
          {items.map((it) => (
            <li key={it.id} className="glass card shadow-sm">
              <div className="text-sm text-slate-400 flex items-center gap-2 mb-1">
                <Calendar className="h-4 w-4 text-blue-300" />
                {new Date(it.date).toLocaleString()}
              </div>
              <div className="font-medium text-blue-300">
                {it.building} • {it.course}
              </div>
              <div className="text-sm text-slate-400">
                Seats: {it.seats?.join(", ")}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
