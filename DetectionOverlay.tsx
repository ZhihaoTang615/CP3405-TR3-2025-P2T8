import React from "react";
import type { Detection } from "@/types";

type Props = {
  /** 最终渲染区域宽高（图片在页面上的真实尺寸）*/
  width: number;
  height: number;
  /** 原图宽高（用于当 detection 是“像素坐标”时换算）*/
  srcW: number;
  srcH: number;
  detections: Detection[];
};

export default function DetectionOverlay({
  width,
  height,
  srcW,
  srcH,
  detections,
}: Props) {
  if (!width || !height) return null;

  // 若检测是“像素坐标”，需要根据原图尺寸 -> 渲染尺寸做比例换算
  const sx = srcW ? width / srcW : 1;
  const sy = srcH ? height / srcH : 1;

  return (
    <svg
      className="pointer-events-none absolute inset-0"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      {detections.map((d, i) => {
        // 兼容两种输入：
        // - 标准化(0-1)：直接乘以渲染尺寸
        // - 像素坐标（>1）：按比例缩放
        const isNormalized =
          d.x <= 1 && d.y <= 1 && d.w <= 1 && d.h <= 1 && d.w > 0 && d.h > 0;

        const x = isNormalized ? d.x * width : d.x * sx;
        const y = isNormalized ? d.y * height : d.y * sy;
        const w = isNormalized ? d.w * width : d.w * sx;
        const h = isNormalized ? d.h * height : d.h * sy;

        const label = `${d.label ?? "Seat"} ${Math.round(
          (d.score ?? 0) * 100
        )}%`;

        return (
          <g key={i}>
            {/* 外框 */}
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill="none"
              stroke="#4f46e5"
              strokeWidth="3"
              rx="8"
            />
            {/* 顶部标签底板 */}
            <rect
              x={x}
              y={Math.max(0, y - 22)}
              width={Math.max(80, w * 0.6)}
              height={22}
              rx={6}
              fill="#4f46e5"
              opacity={0.95}
            />
            {/* 文本 */}
            <text
              x={x + 8}
              y={Math.max(14, y - 7)}
              fill="#fff"
              fontSize="12"
              fontWeight={700}
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
