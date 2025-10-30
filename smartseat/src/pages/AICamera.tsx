// src/pages/AICamera.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Camera as CamIcon,
  Upload,
  Play,
  Square,
  RefreshCw,
} from "lucide-react";
import DetectionOverlay from "@/components/DetectionOverlay";
import type { Detection } from "@/types";
import { mockDetect, yoloDetect } from "@/lib/detect";

type Mode = "mock" | "yolo";

export default function AICamera() {
  const [mode, setMode] = useState<Mode>("mock");
  const [imgURL, setImgURL] = useState("");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);

  const imgRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoOn, setVideoOn] = useState(false);

  const [srcW, setSrcW] = useState(0);
  const [srcH, setSrcH] = useState(0);
  const [renderW, setRenderW] = useState(0);
  const [renderH, setRenderH] = useState(0);

  /** 📁 上传 */
  const onFile = (file: File) => {
    if (/heic|heif/i.test(file.type) || /\.heic$/i.test(file.name)) {
      alert("HEIC/HEIF 在浏览器中可能无法预览，请先转换为 JPG/PNG 再试。");
      return;
    }
    const url = URL.createObjectURL(file);
    setImgURL(url);
    setDetections([]);
  };

  /** 🖱 拖拽上传 */
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files?.[0]) onFile(e.dataTransfer.files[0]);
  };

  /** 🎥 打开/关闭相机 */
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setVideoOn(true);
        console.log("📷 Camera started.");
      }
    } catch (err) {
      console.error("❌ Failed to start camera:", err);
      alert("无法启动摄像头：请检查浏览器权限/是否有可用摄像头。");
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream)
        .getTracks()
        .forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setVideoOn(false);
    console.log("🛑 Camera stopped.");
  };

  /** 📸 捕获帧为图片 */
  const capture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const v = videoRef.current;
    const c = canvasRef.current;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(v, 0, 0, c.width, c.height);
    const dataUrl = c.toDataURL("image/png");
    setImgURL(dataUrl);
    setDetections([]);
    console.log("📸 Captured frame → preview ready.");
  };

  /** 🧠 ✅ 运行 AI 检测（已替换） */
  const runDetect = async () => {
    console.log("🚀 runDetect pressed. imgURL =", imgURL);

    const img = imgRef.current;
    if (!img) {
      console.warn("⚠️ No <img> element, cannot detect.");
      return;
    }

    if (!imgURL) {
      console.warn("⚠️ No image loaded.");
      return;
    }

    console.log(`✅ Mode = ${mode}`);
    setLoading(true);

    try {
      let result: Detection[] = [];

      if (mode === "mock") {
        console.log("🟣 Using Mock AI");
        result = await mockDetect(img);
      } else {
        console.log("🟡 YOLO will run here soon");
        result = await yoloDetect(img);
      }

      console.log("📦 Detection output:", result);
      setDetections(result);
    } catch (err) {
      console.error("❌ Detection error:", err);
    } finally {
      setLoading(false);
    }
  };

  /** 📐 图片加载完，同步尺寸 */
  const handleImgLoad = () => {
    const el = imgRef.current!;
    setSrcW(el.naturalWidth);
    setSrcH(el.naturalHeight);
    setRenderW(el.clientWidth);
    setRenderH(el.clientHeight);

    console.log("🖼 Image loaded:", {
      natural: [el.naturalWidth, el.naturalHeight],
      client: [el.clientWidth, el.clientHeight],
    });
  };

  useEffect(() => {
    if (!imgRef.current) return;
    const el = imgRef.current;
    const ro = new ResizeObserver(() => {
      setRenderW(el.clientWidth);
      setRenderH(el.clientHeight);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [imgURL]);

  return (
    <div className="max-w-5xl mx-auto">
      <header className="mb-6 text-center">
        <h1 className="text-3xl font-bold text-indigo-700">
          🧠 AI Seat Detection
        </h1>
        <p className="text-slate-600 mt-2">Upload / Capture → Run AI</p>
      </header>

      <div className="flex flex-wrap items-center gap-2 mb-4">
        <label className="inline-flex items-center gap-2 border rounded-md px-3 py-2">
          <input
            type="radio"
            checked={mode === "mock"}
            onChange={() => setMode("mock")}
          />
          Mock Demo
        </label>
        <label className="inline-flex items-center gap-2 border rounded-md px-3 py-2">
          <input
            type="radio"
            checked={mode === "yolo"}
            onChange={() => setMode("yolo")}
          />
          YOLO (soon)
        </label>

        <button
          onClick={runDetect}
          disabled={!imgURL || loading}
          className="inline-flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-md disabled:opacity-60"
        >
          {loading ? (
            <RefreshCw className="animate-spin w-4 h-4" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          Run Detection
        </button>
      </div>

      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className="rounded-xl border-2 border-dashed p-6 text-center mb-4"
      >
        <Upload className="w-5 h-5 text-slate-500" />
        <p>Drag & drop / click</p>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
      </div>

      <div className="rounded-xl border bg-white p-4 mb-6">
        {!videoOn ? (
          <button
            onClick={startCamera}
            className="inline-flex items-center gap-2 border px-3 py-2 rounded-md"
          >
            <CamIcon className="w-4 h-4" /> Start Camera
          </button>
        ) : (
          <>
            <div className="flex items-center gap-2">
              <button
                onClick={capture}
                className="inline-flex items-center gap-2 border px-3 py-2 rounded-md"
              >
                📸 Capture
              </button>
              <button
                onClick={stopCamera}
                className="inline-flex items-center gap-2 border px-3 py-2 rounded-md"
              >
                <Square className="w-4 h-4" /> Stop
              </button>
            </div>
            <video
              ref={videoRef}
              className="w-full max-h-[360px] bg-black rounded-md mt-2"
            />
          </>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {imgURL ? (
        <div className="relative border rounded-xl p-3 bg-white">
          <img
            ref={imgRef}
            src={imgURL}
            alt="preview"
            onLoad={handleImgLoad}
            className="max-h-[520px] w-full object-contain rounded-md"
          />
          <DetectionOverlay
            width={renderW}
            height={renderH}
            srcW={srcW}
            srcH={srcH}
            detections={detections}
          />
        </div>
      ) : (
        <div className="rounded-xl border bg-white p-8 text-center text-slate-500">
          Upload or capture to start
        </div>
      )}
    </div>
  );
}
