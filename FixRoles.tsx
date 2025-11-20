// src/pages/FixRoles.tsx
import { useState } from "react";
import { db } from "@/firebase";
import { collection, getDocs, doc, updateDoc } from "firebase/firestore";

export default function FixRoles() {
  const [status, setStatus] = useState("Ready to fix missing roles…");
  const [running, setRunning] = useState(false);

  // ✅ 这里是你的 Lecturer 白名单
  const lecturerWhitelist = ["tanglaoshi666@qq.com"];

  const handleFix = async () => {
    try {
      setRunning(true);
      setStatus("Working… Scanning users collection…");

      const usersRef = collection(db, "users");
      const snap = await getDocs(usersRef);

      let updatedCount = 0;

      for (const docSnap of snap.docs) {
        const data = docSnap.data() as any;
        const email: string = (data.email || "").toLowerCase();
        const currentRole = data.role;
        const userRef = doc(db, "users", docSnap.id);

        // 1️⃣ 已经有 role 的，完全跳过（不覆盖、不修改）
        if (
          currentRole === "admin" ||
          currentRole === "lecturer" ||
          currentRole === "student"
        ) {
          continue;
        }

        // 2️⃣ 没有 role 的，而且在白名单里 → 设为 lecturer
        if (lecturerWhitelist.includes(email)) {
          await updateDoc(userRef, { role: "lecturer" });
          updatedCount++;
          continue;
        }

        // 3️⃣ 其他没有 role 的 → 一律补成 student
        await updateDoc(userRef, { role: "student" });
        updatedCount++;
      }

      setStatus(`✅ Done! Fixed ${updatedCount} user(s).`);
    } catch (err) {
      console.error(err);
      setStatus("❌ Something went wrong. Check console for details.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="mx-auto max-w-xl p-8 space-y-4">
      <h1 className="text-2xl font-bold text-blue-300 mb-2">
        Fix Missing User Roles
      </h1>
      <p className="text-sm text-slate-300">
        This tool will:
        <br />• Keep existing <b>admin / lecturer / student</b> roles unchanged
        <br />• Set any user in whitelist ({lecturerWhitelist.join(", ")}) to{" "}
        <b>lecturer</b> if they don't have a role yet
        <br />• Set all other users without role to <b>student</b>
      </p>

      <button
        onClick={handleFix}
        disabled={running}
        className="btn-primary px-4 py-2 rounded-md disabled:opacity-60"
      >
        {running ? "Running…" : "Run Role Migration Once"}
      </button>

      <p className="text-sm text-slate-400 mt-2">{status}</p>
    </div>
  );
}
