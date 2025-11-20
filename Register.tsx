// src/pages/Register.tsx
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { auth, db } from "@/firebase";
import { doc, setDoc } from "firebase/firestore";
import { createUserWithEmailAndPassword } from "firebase/auth";

export default function Register() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"student" | "lecturer">("student");
  const [error, setError] = useState("");

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      const userCred = await createUserWithEmailAndPassword(
        auth,
        email,
        password
      );
      const user = userCred.user;

      await setDoc(doc(db, "users", user.uid), {
        email: user.email,
        role,
        createdAt: new Date().toISOString(),
      });

      // 注册完默认去学生主流程（lecturer 后面从 navbar 也有自己入口）
      navigate("/reserve");
    } catch (err) {
      console.error(err);
      setError("Registration failed (Email may already exist)");
    }
  };

  return (
    <div className="flex justify-center items-start bg-[var(--bg)] route-fade min-h-screen pt-20 sm:pt-24">
      <div className="glass card w-full max-w-md p-8 animate-fadeIn login-float">
        <h2 className="text-3xl font-bold text-center mb-6 text-blue-300">
          Register
        </h2>

        <form onSubmit={handleRegister} className="space-y-4">
          <input
            type="email"
            placeholder="Email"
            className="input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <input
            type="password"
            placeholder="Password (min 6 chars)"
            className="input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <select
            className="input"
            value={role}
            onChange={(e) => setRole(e.target.value as "student" | "lecturer")}
          >
            <option value="student">Student</option>
            <option value="lecturer">Lecturer</option>
          </select>

          {error && <p className="text-red-400 text-sm text-center">{error}</p>}

          <button
            type="submit"
            className="btn-primary w-full py-2.5 rounded-md"
          >
            Register
          </button>
        </form>

        <p className="text-sm text-center mt-4 text-slate-400">
          Already have an account?{" "}
          <Link
            to="/login"
            className="text-blue-400 hover:underline hover:text-blue-300 transition"
          >
            Login
          </Link>
        </p>
      </div>
    </div>
  );
}
