// src/pages/AICamera.tsx
import React, { useRef, useState, useEffect } from "react";
import { Upload, Camera, Play, Square, RefreshCw } from "lucide-react";
import DetectionOverlay from "@/components/DetectionOverlay";
import type { Detection } from "@/types";

import { mockDetect } from "@/lib/detect"; // ✅ 只保留 mock
import { loadYOLO, yoloDetect } from "@/lib/yoloWeb"; // ✅ YOLO 从 yoloWeb.ts 来

type Mode = "mock" | "yolo";

export default function AICamera() {
  const [mode, setMode] = useState<Mode>("mock");
  const [imgURL, setImgURL] = useState("");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [videoOn, setVideoOn] = useState(false);

  const imgRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [srcW, setSrcW] = useState(0);
  const [srcH, setSrcH] = useState(0);
  const [renderW, setRenderW] = useState(0);
  const [renderH, setRenderH] = useState(0);

  const onFile = (file: File) => {
    const url = URL.createObjectURL(file);
    setImgURL(url);
    setDetections([]);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();
      setVideoOn(true);
    } catch {
      alert("无法访问摄像头，请检查权限");
    }
  };

  const stopCamera = () => {
    (videoRef.current?.srcObject as MediaStream)
      ?.getTracks()
      .forEach((t) => t.stop());
    videoRef.current!.srcObject = null;
    setVideoOn(false);
  };

  const capture = () => {
    const v = videoRef.current!;
    const c = canvasRef.current!;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0, c.width, c.height);

    setImgURL(c.toDataURL("image/png"));
    setDetections([]);
  };

  const runDetect = async () => {
    if (!imgRef.current || !imgURL) return;
    setLoading(true);

    try {
      let result: Detection[] = [];

      if (mode === "mock") {
        result = await mockDetect(imgRef.current);
      } else {
        console.log("🚀 Loading YOLO...");
        await loadYOLO((msg) => console.log(msg));
        result = await yoloDetect(imgRef.current);
      }

      console.log("✅ Detections:", result);
      setDetections(result);
    } catch (e) {
      console.error("❌ YOLO Error:", e);
      alert("AI 检测失败，请查看控制台");
    } finally {
      setLoading(false);
    }
  };

  const handleImgLoad = () => {
    const el = imgRef.current!;
    setSrcW(el.naturalWidth);
    setSrcH(el.naturalHeight);
    setRenderW(el.clientWidth);
    setRenderH(el.clientHeight);
  };

  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setRenderW(el.clientWidth);
      setRenderH(el.clientHeight);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [imgURL]);

  return (
    <div className="max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold text-indigo-700 mb-4">
        🧠 AI Seat Detection
      </h1>

      <div className="flex gap-2 mb-4">
        <label className="inline-flex items-center gap-2 border px-3 py-2 rounded-md">
          <input
            type="radio"
            checked={mode === "mock"}
            onChange={() => setMode("mock")}
          />
          Mock Demo
        </label>

        <label className="inline-flex items-center gap-2 border px-3 py-2 rounded-md">
          <input
            type="radio"
            checked={mode === "yolo"}
            onChange={() => setMode("yolo")}
          />
          YOLO
        </label>

        <button
          onClick={runDetect}
          disabled={!imgURL || loading}
          className="bg-indigo-600 text-white px-4 py-2 rounded-md flex items-center gap-2"
        >
          {loading ? <RefreshCw className="animate-spin" /> : <Play />}
          Run Detection
        </button>
      </div>

      {/* Upload */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className="border-2 border-dashed p-6 text-center rounded-xl mb-4"
      >
        <Upload className="mx-auto mb-2" />
        <p>Drag & drop / click</p>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
      </div>

      {/* Camera */}
      <div className="border p-4 rounded-xl mb-6">
        {!videoOn ? (
          <button
            onClick={startCamera}
            className="border px-3 py-2 rounded-md flex gap-2"
          >
            <Camera /> Start Camera
          </button>
        ) : (
          <>
            <button
              onClick={capture}
              className="border px-3 py-2 rounded-md mr-2"
            >
              📸 Capture
            </button>
            <button
              onClick={stopCamera}
              className="border px-3 py-2 rounded-md"
            >
              <Square /> Stop
            </button>
            <video
              ref={videoRef}
              className="w-full max-h-[320px] bg-black mt-2 rounded-md"
            />
          </>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Preview + overlay */}
      {imgURL && (
        <div className="relative border rounded-xl p-3 bg-white">
          <img
            ref={imgRef}
            src={imgURL}
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
      )}
    </div>
  );
}
