// src/components/Navbar.tsx
import React, { useState } from "react";
import { NavLink, Link } from "react-router-dom";
import {
  LayoutDashboard,
  Map,
  CalendarCheck2,
  Info,
  Menu,
  X,
  BarChart3,
  Camera,
} from "lucide-react";

const linkBase =
  "inline-flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200";
const cls = (active: boolean) =>
  active
    ? `${linkBase} bg-indigo-600 text-white shadow-sm`
    : `${linkBase} text-slate-600 hover:text-indigo-700 hover:bg-indigo-50`;

export default function Navbar() {
  const [open, setOpen] = useState(false);

  const LinkItem = ({
    to,
    end,
    children,
    icon,
  }: {
    to: string;
    end?: boolean;
    children: React.ReactNode;
    icon: React.ReactNode;
  }) => (
    <NavLink to={to} end={end} className={({ isActive }) => cls(isActive)}>
      {icon}
      {children}
    </NavLink>
  );

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200 bg-white/80 backdrop-blur-md shadow-sm">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <Link
          to="/"
          className="flex items-center gap-2 font-semibold text-slate-900 hover:text-indigo-700 transition"
        >
          <span className="text-xl">🎯 SmartSeat</span>
        </Link>

        {/* Desktop */}
        <nav className="hidden md:flex items-center gap-2">
          <LinkItem to="/" end icon={<LayoutDashboard className="size-4" />}>
            Home
          </LinkItem>
          <LinkItem to="/seats" icon={<Map className="size-4" />}>
            Seat Map
          </LinkItem>
          <LinkItem to="/booking" icon={<CalendarCheck2 className="size-4" />}>
            Booking
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

          {/* ✅ AI Vision link */}
          <LinkItem to="/ai-camera" icon={<Camera className="size-4" />}>
            AI Vision
          </LinkItem>

          <LinkItem to="/about" icon={<Info className="size-4" />}>
            About
          </LinkItem>
        </nav>

        {/* Mobile button */}
        <button
          className="md:hidden p-2 rounded-md hover:bg-slate-100 transition"
          onClick={() => setOpen(!open)}
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile */}
      {open && (
        <nav className="md:hidden bg-white border-t border-slate-200 shadow-lg">
          <div className="flex flex-col items-start px-4 py-2 space-y-1">
            <NavLink
              to="/"
              end
              className={({ isActive }) => cls(isActive)}
              onClick={() => setOpen(false)}
            >
              <LayoutDashboard className="size-4" /> Home
            </NavLink>
            <NavLink
              to="/seats"
              className={({ isActive }) => cls(isActive)}
              onClick={() => setOpen(false)}
            >
              <Map className="size-4" /> Seat Map
            </NavLink>
            <NavLink
              to="/booking"
              className={({ isActive }) => cls(isActive)}
              onClick={() => setOpen(false)}
            >
              <CalendarCheck2 className="size-4" /> Booking
            </NavLink>
            <NavLink
              to="/reservations"
              className={({ isActive }) => cls(isActive)}
              onClick={() => setOpen(false)}
            >
              <CalendarCheck2 className="size-4" /> My Reservations
            </NavLink>
            <NavLink
              to="/dashboard"
              className={({ isActive }) => cls(isActive)}
              onClick={() => setOpen(false)}
            >
              <BarChart3 className="size-4" /> Dashboard
            </NavLink>

            {/* ✅ AI Vision in mobile */}
            <NavLink
              to="/ai-camera"
              className={({ isActive }) => cls(isActive)}
              onClick={() => setOpen(false)}
            >
              <Camera className="size-4" /> AI Vision
            </NavLink>

            <NavLink
              to="/about"
              className={({ isActive }) => cls(isActive)}
              onClick={() => setOpen(false)}
            >
              <Info className="size-4" /> About
            </NavLink>
          </div>
        </nav>
      )}
    </header>
  );
}
