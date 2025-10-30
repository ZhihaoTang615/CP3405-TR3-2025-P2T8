// src/App.tsx
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import SeatMap from "./pages/SeatMap";
import Booking from "./pages/Booking";
import About from "./pages/About";
import MyReservations from "./pages/MyReservations";
import Dashboard from "./pages/Dashboard";
import AICamera from "./pages/AICamera";

function Footer() {
  return (
    <footer className="border-t mt-12">
      <div className="container mx-auto text-sm text-gray-500 py-8 px-4">
        © {new Date().getFullYear()} SmartSeat • Built with React + Tailwind
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <Navbar />
      <main className="container mx-auto py-10 px-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/seats" element={<SeatMap />} />
          <Route path="/booking" element={<Booking />} />
          <Route path="/reservations" element={<MyReservations />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/about" element={<About />} />

          {/* ✅ AI Vision Seat Detection */}
          <Route path="/ai-camera" element={<AICamera />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}
