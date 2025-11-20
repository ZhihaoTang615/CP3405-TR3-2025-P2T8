// src/lib/detect.ts
import type { Detection } from "@/types";
import * as ort from "onnxruntime-web";

/** ✅ Mock detection — only for demo */
export async function mockDetect(img: HTMLImageElement): Promise<Detection[]> {
  await new Promise((r) => setTimeout(r, 700));
  return [
    { x: 0.2, y: 0.3, w: 0.15, h: 0.22, score: 0.91, label: "Seat" },
    { x: 0.55, y: 0.4, w: 0.18, h: 0.25, score: 0.87, label: "Seat" },
  ];
}

/** Load ONNX session */
const sessionPromise = ort.InferenceSession.create("/models/yolov8n.onnx", {
  executionProviders: ["wasm"],
});

/** Real YOLO detect */
export async function yoloDetect(img: HTMLImageElement): Promise<Detection[]> {
  const session = await sessionPromise;
  const { data, width, height } = imageToTensor(img);
  const inputTensor = new ort.Tensor("float32", data, [1, 3, height, width]);

  const results = await session.run({ images: inputTensor });
  const output = results.output0.data;

  return parseYOLO(output);
}

function imageToTensor(img: HTMLImageElement) {
  const size = 640;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, size, size);

  const imgData = ctx.getImageData(0, 0, size, size);
  const data = new Float32Array(3 * size * size);
  let idx = 0;

  for (let i = 0; i < imgData.data.length; i += 4) {
    data[idx++] = imgData.data[i] / 255;
    data[idx++] = imgData.data[i + 1] / 255;
    data[idx++] = imgData.data[i + 2] / 255;
  }
  return { data, width: size, height: size };
}

/** Parse YOLO detection results */
function parseYOLO(arr: any[]): Detection[] {
  const dets: Detection[] = [];
  const numDet = arr.length / 84;

  for (let i = 0; i < numDet; i++) {
    const o = i * 84;
    const score = arr[o + 4];
    if (score > 0.5) {
      const x = arr[o],
        y = arr[o + 1],
        w = arr[o + 2],
        h = arr[o + 3];
      dets.push({
        x: x - w / 2,
        y: y - h / 2,
        w,
        h,
        score,
        label: "Seat",
      });
    }
  }
  return dets;
}
