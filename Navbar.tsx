// src/components/Navbar.tsx
import React, { useState, useEffect } from "react";
import { NavLink, Link, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  CalendarCheck2,
  BarChart3,
  Menu,
  X,
  Sun,
  Moon,
  Users,
  ClipboardList,
} from "lucide-react";
import { useAuth } from "@/lib/auth";

const linkBase =
  "relative inline-flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all";

const cls = (active: boolean, theme: string) =>
  theme === "light"
    ? active
      ? `${linkBase} text-blue-700`
      : `${linkBase} text-slate-800 hover:text-blue-700`
    : active
    ? `${linkBase} text-blue-300`
    : `${linkBase} text-slate-300 hover:text-blue-200`;

function ActiveUnderline({ active }: { active: boolean }) {
  return (
    <span
      className={`pointer-events-none absolute -bottom-1 left-3 right-3 h-[2px] rounded-full transition ${
        active
          ? "bg-[linear-gradient(90deg,#1e3a8a,#6366f1)] opacity-100"
          : "opacity-0"
      }`}
    />
  );
}

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");

  const { user, role, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const html = document.documentElement;
    html.classList.remove("light", "dark");
    html.classList.add(theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(theme === "dark" ? "light" : "dark");
  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  const LinkItem = ({ to, end, children, icon }: any) => (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) => cls(isActive, theme)}
    >
      {({ isActive }) => (
        <>
          {icon}
          {children}
          <ActiveUnderline active={isActive} />
        </>
      )}
    </NavLink>
  );

  return (
    <header className="sticky top-0 z-40 w-full navbar-glass">
      <div className="h-[2px] w-full bg-[linear-gradient(90deg,#0ea5e9,#6366f1,#0ea5e9)] opacity-60" />

      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        {/* Logo */}
        <Link
          to="/"
          className={`flex items-center gap-2 font-semibold ${
            theme === "light" ? "text-slate-900" : "text-blue-300"
          }`}
        >
          <span className="text-xl">🎯 SmartSeat</span>
        </Link>

        {/* Desktop navigation */}
        <nav className="hidden md:flex items-center gap-3">
          {!user && (
            <>
              <LinkItem to="/login">Login</LinkItem>
              <LinkItem to="/register">Register</LinkItem>
            </>
          )}

          {/* STUDENT */}
          {user && role === "student" && (
            <>
              <LinkItem
                to="/home"
                end
                icon={<LayoutDashboard className="size-4" />}
              >
                Home
              </LinkItem>
              <LinkItem
                to="/reserve"
                icon={<CalendarCheck2 className="size-4" />}
              >
                Reserve
              </LinkItem>
              <LinkItem
                to="/reservations"
                icon={<CalendarCheck2 className="size-4" />}
              >
                My Reservations
              </LinkItem>
              <LinkItem to="/dashboard" icon={<BarChart3 className="size-4" />}>
                Dashboard
              </LinkItem>
            </>
          )}

          {/* LECTURER */}
          {user && role === "lecturer" && (
            <>
              <LinkItem
                to="/lecturer"
                end
                icon={<ClipboardList className="size-4" />}
              >
                Lecturer Dashboard
              </LinkItem>

              {/* FIXED PATH - from /reserve -> /lecturer/reserve */}
              <LinkItem
                to="/lecturer/reserve"
                icon={<CalendarCheck2 className="size-4" />}
              >
                Reserve
              </LinkItem>

              <LinkItem
                to="/lecturer/reservations"
                icon={<CalendarCheck2 className="size-4" />}
              >
                All Reservations
              </LinkItem>
            </>
          )}

          {/* ADMIN */}
          {user && role === "admin" && (
            <>
              <LinkItem
                to="/admin"
                end
                icon={<LayoutDashboard className="size-4" />}
              >
                Admin Dashboard
              </LinkItem>
              <LinkItem to="/admin/users" icon={<Users className="size-4" />}>
                Users
              </LinkItem>
              <LinkItem
                to="/admin/reservations"
                icon={<CalendarCheck2 className="size-4" />}
              >
                Reservations
              </LinkItem>
            </>
          )}

          {user && (
            <span
              className={`text-sm ${
                theme === "light" ? "text-slate-800" : "text-slate-300"
              }`}
            >
              👤 {user.email}
              <span className="ml-1 text-xs uppercase text-slate-400">
                ({role})
              </span>
            </span>
          )}

          {user && (
            <button
              onClick={handleLogout}
              className="btn-ghost rounded-md px-3 py-1.5 hover:bg-blue-500/20"
            >
              Logout
            </button>
          )}

          <button
            onClick={toggleTheme}
            className="p-2 rounded-md hover:bg-blue-500/20 transition ml-1"
          >
            {theme === "dark" ? (
              <Sun className="h-5 w-5 text-yellow-300" />
            ) : (
              <Moon className="h-5 w-5 text-blue-600" />
            )}
          </button>
        </nav>

        {/* Mobile Menu Toggle */}
        <button
          className="md:hidden p-2 rounded-md text-white/90 hover:bg-white/10"
          onClick={() => setOpen(!open)}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden flex flex-col px-4 pb-4 space-y-2 bg-black/40 backdrop-blur-xl border-t border-white/10">
          {user && role === "lecturer" && (
            <>
              <LinkItem to="/lecturer" end>
                Lecturer Dashboard
              </LinkItem>
              <LinkItem to="/lecturer/reserve">Reserve</LinkItem>
              <LinkItem to="/lecturer/reservations">All Reservations</LinkItem>
            </>
          )}

          {user && role === "admin" && (
            <>
              <LinkItem to="/admin" end>
                Admin Dashboard
              </LinkItem>
              <LinkItem to="/admin/users">Users</LinkItem>
              <LinkItem to="/admin/reservations">Reservations</LinkItem>
            </>
          )}

          {user && (
            <button
              onClick={handleLogout}
              className="text-left px-3 py-2 text-red-400 hover:text-red-300"
            >
              Logout
            </button>
          )}
        </div>
      )}
    </header>
  );
}
