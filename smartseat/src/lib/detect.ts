// src/lib/detect.ts
import type { Detection } from "@/types";

/** ✅ Mock detection：标准化(0-1)坐标，便于任何分辨率都能正确 Overlay */
export async function mockDetect(_img: HTMLImageElement): Promise<Detection[]> {
  // 模拟“AI 思考”时间
  await new Promise((r) => setTimeout(r, 800));

  return [
    { x: 0.2, y: 0.3, w: 0.15, h: 0.22, score: 0.95, label: "Seat" },
    { x: 0.55, y: 0.4, w: 0.18, h: 0.25, score: 0.91, label: "Seat" },
  ];
}

/** TODO: 真 YOLO 浏览器推理；现在仅做占位 */
export async function yoloDetect(_img: HTMLImageElement): Promise<Detection[]> {
  await new Promise((r) => setTimeout(r, 800));
  return [];
}
