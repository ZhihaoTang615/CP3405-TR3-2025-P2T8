// src/pages/Login.tsx
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { login } from "@/lib/auth";
import { db } from "@/firebase";
import { doc, getDoc } from "firebase/firestore";

export default function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      const userCred = await login(email, password);
      const user = userCred.user;

      const ref = doc(db, "users", user.uid);
      const snap = await getDoc(ref);

      let role: string = "student";
      if (snap.exists()) {
        role = (snap.data() as any).role || "student";
      }

      if (role === "admin") navigate("/admin");
      else if (role === "lecturer") navigate("/lecturer");
      else navigate("/reserve");
    } catch (err) {
      console.error(err);
      setError("Invalid email or password");
    }
  };

  return (
    <div className="flex justify-center items-start bg-[var(--bg)] min-h-screen pt-20">
      <div className="glass card w-full max-w-md p-8 animate-fadeIn">
        <h2 className="text-3xl font-bold text-center mb-6 text-blue-300">
          Login
        </h2>

        <form onSubmit={handleLogin} className="space-y-4">
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
            placeholder="Password"
            className="input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          {error && <p className="text-red-400 text-sm text-center">{error}</p>}

          <button
            type="submit"
            className="btn-primary w-full py-2.5 rounded-md"
          >
            Login
          </button>
        </form>

        <p className="text-sm text-center mt-4 text-slate-400">
          Don’t have an account?{" "}
          <Link
            to="/register"
            className="text-blue-400 hover:underline hover:text-blue-300 transition"
          >
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
