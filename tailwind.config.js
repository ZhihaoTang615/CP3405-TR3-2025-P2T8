/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // 🎨 SmartSeat 主色调
        primary: "#3B82F6", // 主蓝色（按钮、链接）
        secondary: "#1E293B", // 深蓝灰（背景/标题）
        accent: "#22C55E", // 成功/激活色
        danger: "#EF4444", // 错误提示色
        warning: "#FACC15", // 警告色
        neutral: "#F8FAFC", // 页面背景
      },
      fontFamily: {
        // 🖋 字体风格
        sans: [
          "Inter",
          "system-ui",
          "Avenir",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        heading: ["Poppins", "Helvetica", "Arial", "sans-serif"],
      },
      boxShadow: {
        soft: "0 4px 12px rgba(0, 0, 0, 0.1)", // 柔和阴影
      },
      borderRadius: {
        xl: "1rem",
      },
    },
  },
  plugins: [],
};
