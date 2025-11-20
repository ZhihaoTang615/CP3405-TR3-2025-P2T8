export type SeatStatus = "free" | "reserved" | "mine";

export type Seat = {
  id: string;
  status: SeatStatus;
};

export type User = {
  name: string;
  email: string;
};

export type Reservation = {
  id: number;
  user: User;
  seats: string[];
  building: string;
  course: string;
  date: string; // ISO date
  createdAt: string;
};

/** ✅ 统一的检测框类型（与 Overlay 对齐）*/
export type Detection = {
  /** 左上角 x，通常是相对(0-1)，也可为原图像素（自动适配） */
  x: number;
  /** 左上角 y */
  y: number;
  /** 宽度 */
  w: number;
  /** 高度 */
  h: number;
  /** 置信度 0-1 */
  score: number;
  /** 类别名 */
  label: string;
};
