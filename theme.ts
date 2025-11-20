// src/lib/theme.ts
export function initTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "dark";
  const saved = localStorage.getItem("theme") as "light" | "dark" | null;
  if (saved) {
    document.documentElement.classList.add(saved);
    return saved;
  }
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = prefersDark ? "dark" : "light";
  document.documentElement.classList.add(theme);
  return theme;
}

export function applyTheme(theme: "light" | "dark") {
  const html = document.documentElement;
  html.classList.remove("light", "dark");
  html.classList.add(theme);
  localStorage.setItem("theme", theme);
}
