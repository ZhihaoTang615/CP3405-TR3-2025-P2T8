import React from "react";
import { Link } from "react-router-dom";

const About: React.FC = () => {
  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 animate-fadeIn">
        <section className="mx-auto my-12 text-center">
          <h1 className="bg-gradient-to-r from-slate-900 to-indigo-700 bg-clip-text text-4xl font-extrabold tracking-tight text-transparent sm:text-5xl">
            👋 About SmartSeat
          </h1>
          <p className="mt-4 text-slate-600 text-lg max-w-2xl mx-auto">
            SmartSeat leverages AI to help students, teams, and organizations
            find their optimal seating arrangement—balancing comfort,
            collaboration, and focus.
          </p>
        </section>

        <section className="grid gap-6 md:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm hover:-translate-y-1 hover:shadow-lg transition">
            <h3 className="text-lg font-semibold text-slate-900">
              Why SmartSeat?
            </h3>
            <ul className="mt-3 list-disc pl-5 text-slate-700 leading-7">
              <li>Realtime seat status and environment data</li>
              <li>AI-based seat ranking and personalized suggestions</li>
              <li>One-click booking with instant confirmation</li>
              <li>Cross-room compatibility for classrooms and offices</li>
            </ul>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm hover:-translate-y-1 hover:shadow-lg transition">
            <h3 className="text-lg font-semibold text-slate-900">
              Team & Contact
            </h3>
            <p className="mt-3 text-slate-700 leading-7">
              Built with ❤️ by the SmartSeat Team. For inquiries or
              collaboration, email us at{" "}
              <span className="font-medium text-indigo-600">
                support@smartseat.app
              </span>
              .
            </p>
            <div className="mt-4 text-sm text-slate-500">Version: 1.0.0</div>
          </div>
        </section>

        <div className="my-12 text-center">
          <Link
            to="/"
            className="rounded-lg border border-slate-300 bg-white px-6 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition hover:-translate-y-0.5 hover:bg-slate-50 hover:shadow"
          >
            Back to Home
          </Link>
        </div>
      </main>
    </div>
  );
};

export default About;
