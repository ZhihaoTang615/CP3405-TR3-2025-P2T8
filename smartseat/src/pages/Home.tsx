import React from "react";
import { Link } from "react-router-dom";

const Home: React.FC = () => {
  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 animate-fadeIn">
        {/* Hero Section */}
        <section className="mx-auto my-16 max-w-3xl text-center">
          <h1 className="bg-gradient-to-r from-slate-900 to-indigo-700 bg-clip-text text-4xl font-extrabold tracking-tight text-transparent sm:text-5xl">
            🚀 Welcome to SmartSeat
          </h1>
          <p className="mt-4 text-lg leading-7 text-slate-600">
            Use AI to recommend the best seat in class, meeting rooms, and
            events — balancing comfort, collaboration, and performance.
          </p>

          <div className="mt-8 flex items-center justify-center">
            <Link
              to="/seats"
              className="rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg transition-transform duration-200 hover:-translate-y-0.5 hover:bg-indigo-700 hover:shadow-xl active:translate-y-0"
            >
              Explore Seat Map
            </Link>
          </div>
        </section>

        {/* Feature Cards */}
        <section className="grid gap-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm md:grid-cols-3">
          <Feature
            title="Realtime Status"
            desc="See seat occupancy & noise level."
            emoji="📡"
          />
          <Feature
            title="Smart Ranking"
            desc="AI ranks seats by your preferences."
            emoji="🤖"
          />
          <Feature
            title="One-Click Booking"
            desc="Reserve and share in seconds."
            emoji="⚡"
          />
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-12 border-t border-slate-200 bg-white/80 py-6 text-sm text-slate-500 text-center">
        <div>© 2025 SmartSeat • Built with React + Tailwind</div>
      </footer>
    </div>
  );
};

const Feature: React.FC<{ title: string; desc: string; emoji: string }> = ({
  title,
  desc,
  emoji,
}) => (
  <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition-transform duration-200 hover:-translate-y-1 hover:shadow-lg hover:border-indigo-200">
    <div className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-50 text-xl">
      {emoji}
    </div>
    <h3 className="mt-3 text-lg font-semibold text-slate-900">{title}</h3>
    <p className="mt-1.5 text-sm leading-6 text-slate-600">{desc}</p>
  </div>
);

export default Home;
