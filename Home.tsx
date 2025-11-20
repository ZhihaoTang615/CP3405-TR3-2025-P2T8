export default function Home() {
  return (
    <div className="mx-auto max-w-5xl space-y-12 p-6 route-fade">
      <section className="glass card text-center py-12 relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_40%,rgba(79,70,229,0.15),transparent_70%),radial-gradient(circle_at_80%_60%,rgba(14,165,233,0.12),transparent_70%)] pointer-events-none" />
        <div className="relative z-10">
          <h1 className="text-5xl font-bold text-blue-300 mb-3 tracking-tight">
            SmartSeat
          </h1>
          <p className="max-w-2xl mx-auto text-slate-400 leading-relaxed">
            Find a seat in seconds — real-time availability, fair allocation,
            and effortless booking.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-4">
            <a href="/reserve" className="btn-primary px-5 py-2.5 rounded-lg">
              Reserve Now
            </a>
            <a
              href="/dashboard"
              className="btn-ghost px-5 py-2.5 rounded-lg hover:bg-blue-500/20"
            >
              View Dashboard
            </a>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card
          title="Real-time Seats"
          desc="Live seat maps with instant updates by block, floor, or classroom."
          icon="💺"
        />
        <Card
          title="Smart Analytics"
          desc="Understand space usage by building, time, and course."
          icon="📊"
        />
        <Card
          title="AI Vision (Beta)"
          desc="Detect and count seats from camera or uploaded image — right in browser."
          icon="🧠"
        />
      </section>
    </div>
  );
}

function Card({
  title,
  desc,
  icon,
}: {
  title: string;
  desc: string;
  icon: string;
}) {
  return (
    <div className="glass card text-center p-6 hover:scale-[1.02] transition-transform duration-200">
      <div className="text-3xl mb-3">{icon}</div>
      <h3 className="font-semibold text-blue-300 mb-1">{title}</h3>
      <p className="text-sm text-slate-400">{desc}</p>
    </div>
  );
}
