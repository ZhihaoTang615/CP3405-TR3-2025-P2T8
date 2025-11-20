// src/App.tsx
import React from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";

import Home from "./pages/Home";
import MyReservations from "./pages/MyReservations";
import Dashboard from "./pages/Dashboard";
import Reserve from "./pages/Reserve";
import Login from "./pages/Login";
import Register from "./pages/Register";

import LecturerDashboard from "./pages/LecturerDashboard";
import LecturerReservations from "./pages/LecturerReservations";
import LecturerReserve from "./pages/LecturerReserve";

import AdminOverview from "./pages/AdminOverview"; // ✅ 用新的 Overview
import AdminUsers from "./pages/AdminUsers";
import AdminReservations from "./pages/AdminReservations";

import FixRoles from "./pages/FixRoles";
import { useAuth } from "./lib/auth";

// ============== Protected ==============
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="p-8 text-center">Loading…</div>;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function RoleRoute({
  allowed,
  children,
}: {
  allowed: ("student" | "lecturer" | "admin")[];
  children: React.ReactNode;
}) {
  const { user, loading, role } = useAuth();
  if (loading) return <div className="p-8 text-center">Loading…</div>;
  if (!user) return <Navigate to="/login" replace />;
  if (!role || !allowed.includes(role)) return <Navigate to="/" replace />;
  return <>{children}</>;
}

// ============== Footer ==============
function Footer() {
  return (
    <footer className="mt-12">
      <div className="mx-auto max-w-6xl text-sm text-slate-400 py-10 px-4">
        © {new Date().getFullYear()} SmartSeat • React + Tailwind
      </div>
    </footer>
  );
}

// ============== Main App Component ==============
export default function App() {
  const location = useLocation();
  const { user, role } = useAuth();

  return (
    // 🌞 Light = black / 🌙 Dark = pure white
    <div className="min-h-screen text-slate-900 dark:text-white">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        <div key={location.pathname} className="route-fade">
          <Routes location={location}>
            {/* 🏠 Landing → role-based redirect */}
            <Route
              path="/"
              element={
                user ? (
                  role === "admin" ? (
                    <Navigate to="/admin" replace />
                  ) : role === "lecturer" ? (
                    <Navigate to="/lecturer" replace />
                  ) : (
                    <Navigate to="/home" replace />
                  )
                ) : (
                  <Home />
                )
              }
            />

            {/* Public */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />

            {/* FixRoles */}
            <Route
              path="/fix-roles"
              element={
                <ProtectedRoute>
                  <FixRoles />
                </ProtectedRoute>
              }
            />

            {/* Home (student + lecturer) */}
            <Route
              path="/home"
              element={
                <RoleRoute allowed={["student", "lecturer"]}>
                  <Home />
                </RoleRoute>
              }
            />

            {/* Reserve for students only */}
            <Route
              path="/reserve"
              element={
                <RoleRoute allowed={["student"]}>
                  <Reserve />
                </RoleRoute>
              }
            />

            {/* Student only */}
            <Route
              path="/reservations"
              element={
                <RoleRoute allowed={["student"]}>
                  <MyReservations />
                </RoleRoute>
              }
            />
            <Route
              path="/dashboard"
              element={
                <RoleRoute allowed={["student"]}>
                  <Dashboard />
                </RoleRoute>
              }
            />

            {/* Lecturer only */}
            <Route
              path="/lecturer"
              element={
                <RoleRoute allowed={["lecturer"]}>
                  <LecturerDashboard />
                </RoleRoute>
              }
            />
            <Route
              path="/lecturer/reserve"
              element={
                <RoleRoute allowed={["lecturer"]}>
                  <LecturerReserve />
                </RoleRoute>
              }
            />
            <Route
              path="/lecturer/reservations"
              element={
                <RoleRoute allowed={["lecturer"]}>
                  <LecturerReservations />
                </RoleRoute>
              }
            />

            {/* Admin only */}
            <Route
              path="/admin"
              element={
                <RoleRoute allowed={["admin"]}>
                  <AdminOverview /> {/* ✅ 这里用新的总览页 */}
                </RoleRoute>
              }
            />
            <Route
              path="/admin/users"
              element={
                <RoleRoute allowed={["admin"]}>
                  <AdminUsers />
                </RoleRoute>
              }
            />
            <Route
              path="/admin/reservations"
              element={
                <RoleRoute allowed={["admin"]}>
                  <AdminReservations />
                </RoleRoute>
              }
            />

            {/* 兜底 */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>

      <Footer />
    </div>
  );
}
