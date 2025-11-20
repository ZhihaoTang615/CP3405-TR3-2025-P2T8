import React, { useEffect, useState } from "react";
import {
  collection,
  query,
  where,
  getDocs,
  onSnapshot,
} from "firebase/firestore";
import { db } from "@/firebase";
import { useAuth } from "@/lib/auth";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { RefreshCw } from "lucide-react";

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    total: 0,
    seats: 0,
    buildings: 0,
    weekly: [] as { name: string; seats: number }[],
    buildingChart: [] as { name: string; count: number }[],
  });
  const [loading, setLoading] = useState(true);
  const [colorSeed, setColorSeed] = useState(Math.random()); // 用于随机配色

  const loadData = async () => {
    if (!user) return;
    setLoading(true);

    const q = query(
      collection(db, "reservations"),
      where("uid", "==", user.uid)
    );
    const docsSnap = await getDocs(q);
    const reservations = docsSnap.docs.map((d) => d.data());

    const total = reservations.length;
    const seats = reservations.reduce(
      (sum, r: any) => sum + (r.seats?.length || 0),
      0
    );
    const buildingSet = new Set(reservations.map((r: any) => r.building));
    const buildings = buildingSet.size;

    const weekMap: Record<string, number> = {};
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    reservations.forEach((r: any) => {
      const d = new Date(r.date);
      const day = days[(d.getDay() + 6) % 7];
      weekMap[day] = (weekMap[day] || 0) + (r.seats?.length || 0);
    });

    const weekly = days.map((day) => ({
      name: day,
      seats: weekMap[day] || 0,
    }));

    const buildingCount: Record<string, number> = {};
    reservations.forEach((r: any) => {
      buildingCount[r.building] = (buildingCount[r.building] || 0) + 1;
    });
    const buildingChart = Object.entries(buildingCount).map(
      ([name, count]) => ({
        name,
        count: Number(count),
      })
    );

    setStats({ total, seats, buildings, weekly, buildingChart });
    setLoading(false);
  };

  useEffect(() => {
    loadData();
  }, [user]);

  const isDarkMode = document.documentElement.classList.contains("dark");

  const refresh = () => {
    setColorSeed(Math.random()); // 随机新配色
    loadData(); // 重新加载数据
  };

  // 根据随机种子生成颜色
  const randomColor = (i: number, total: number) => {
    const base = (i * 360) / total + colorSeed * 360;
    const hue = base % 360;
    return isDarkMode ? `hsl(${hue}, 70%, 65%)` : `hsl(${hue}, 70%, 55%)`;
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8 p-6 route-fade">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-blue-300">Dashboard</h1>
          <p className="text-slate-400">Your SmartSeat usage analytics</p>
        </div>

        {/* 刷新按钮 */}
        <button
          onClick={refresh}
          className="glass px-3 py-2 flex items-center gap-2 rounded-md hover:bg-white/10 transition active:scale-95"
          title="Refresh Data"
        >
          <RefreshCw className="h-5 w-5 text-blue-400 hover:rotate-180 transition-transform duration-500" />
          <span className="text-slate-300 text-sm">Refresh</span>
        </button>
      </header>

      {/* 数字统计 */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Stat label="Total Reservations" value={stats.total} />
        <Stat label="Seats Booked" value={stats.seats} />
        <Stat label="Buildings Used" value={stats.buildings} />
      </div>

      {/* 图表区 */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* 柱状图 */}
        <ChartCard title="Weekly Seat Usage">
          {loading ? (
            <Skeleton />
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={stats.weekly}>
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{
                    background: "rgba(20,28,48,0.9)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    color: "#ffffff",
                  }}
                  itemStyle={{ color: "#ffffff" }}
                  labelStyle={{ color: "#ffffff" }}
                />
                <Bar dataKey="seats" radius={[6, 6, 0, 0]}>
                  {stats.weekly.map((_, i) => (
                    <Cell key={i} fill={randomColor(i, stats.weekly.length)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartCard>

        {/* 饼图（带数字标签） */}
        <ChartCard title="By Building">
          {loading ? (
            <Skeleton />
          ) : stats.buildingChart.length === 0 ? (
            <div className="h-[260px] grid place-items-center text-slate-500">
              No building data yet
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <PieChart margin={{ top: 20, bottom: 8 }}>
                <Pie
                  data={stats.buildingChart}
                  dataKey="count"
                  nameKey="name"
                  outerRadius={100}
                  label
                  labelLine
                >
                  {stats.buildingChart.map((_, i) => (
                    <Cell
                      key={i}
                      fill={randomColor(i, stats.buildingChart.length)}
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "rgba(20,28,48,0.9)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    color: "#ffffff",
                  }}
                  itemStyle={{ color: "#ffffff" }}
                  labelStyle={{ color: "#ffffff" }}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
        </ChartCard>
      </div>
    </div>
  );
}

/* 📊 小卡片组件 */
function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="glass card">
      <p className="text-sm text-slate-400">{label}</p>
      <div className="mt-1 text-3xl font-semibold text-blue-300">{value}</div>
    </div>
  );
}

/* 🧱 图表容器组件 */
function ChartCard({
  title,
  children,
}: React.PropsWithChildren<{ title: string }>) {
  return (
    <div className="glass card">
      <h3 className="mb-2 font-semibold text-blue-300">{title}</h3>
      {children}
    </div>
  );
}

/* 🔄 加载占位组件 */
function Skeleton() {
  return <div className="h-[260px] rounded-xl bg-slate-800/30 animate-pulse" />;
}
