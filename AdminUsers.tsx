// src/pages/AdminUsers.tsx
import React, { useEffect, useState } from "react";
import {
  collection,
  onSnapshot,
  updateDoc,
  doc,
  deleteDoc,
} from "firebase/firestore";
import { db } from "@/firebase";
import { Trash2, Users } from "lucide-react";

interface UserRow {
  id: string;
  email?: string;
  role?: string;
  createdAt?: string;
}

export default function AdminUsers() {
  const [users, setUsers] = useState<UserRow[]>([]);

  useEffect(() => {
    const unsub = onSnapshot(collection(db, "users"), (snap) => {
      const data = snap.docs.map((d) => ({
        id: d.id,
        ...(d.data() as any),
      }));
      setUsers(data);
    });

    return () => unsub();
  }, []);

  const handleRoleChange = async (id: string, role: string) => {
    await updateDoc(doc(db, "users", id), { role });
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this user record? This does NOT delete auth user.")) {
      return;
    }
    await deleteDoc(doc(db, "users", id));
  };

  return (
    <div className="mx-auto max-w-5xl p-6 space-y-4 text-[var(--text)]">
      <h1 className="text-3xl font-bold flex items-center gap-2 text-[var(--accent)]">
        <Users className="h-7 w-7 text-[var(--accent)]" />
        Manage Users
      </h1>
      <p className="text-sm text-[var(--muted)]">
        Change roles and clean up user records in the system.
      </p>

      <div
        className="
          rounded-2xl 
          glass
          overflow-x-auto p-4
        "
      >
        <table className="min-w-full text-sm text-[var(--text)]">
          <thead className="border-b border-[var(--border)]">
            <tr>
              <th className="text-left py-2 pr-4 font-semibold">Email</th>
              <th className="text-left py-2 pr-4 font-semibold">Role</th>
              <th className="text-left py-2 pr-4 font-semibold">Created At</th>
              <th className="text-left py-2 font-semibold">Actions</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-b border-[var(--border)]">
                <td className="py-2 pr-4">{u.email || "—"}</td>
                <td className="py-2 pr-4">
                  <select
                    className="
                      bg-transparent 
                      border border-[var(--border)]
                      text-[var(--text)]
                      rounded px-2 py-1 text-xs
                    "
                    value={u.role || "student"}
                    onChange={(e) => handleRoleChange(u.id, e.target.value)}
                  >
                    <option value="student">student</option>
                    <option value="lecturer">lecturer</option>
                    <option value="admin">admin</option>
                  </select>
                </td>
                <td className="py-2 pr-4">
                  {u.createdAt ? new Date(u.createdAt).toLocaleString() : "—"}
                </td>
                <td className="py-2">
                  <button
                    onClick={() => handleDelete(u.id)}
                    className="inline-flex items-center gap-1 text-xs text-red-500 hover:text-red-400"
                  >
                    <Trash2 className="h-3 w-3" />
                    Delete
                  </button>
                </td>
              </tr>
            ))}

            {users.length === 0 && (
              <tr>
                <td
                  colSpan={4}
                  className="py-4 text-center text-sm text-[var(--muted)]"
                >
                  No users found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
