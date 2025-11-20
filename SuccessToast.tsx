import React from "react";

export default function SuccessToast({ text }: { text: string }) {
  return (
    <div className="fixed bottom-4 right-4 rounded-md bg-emerald-600 px-3 py-2 text-sm text-white shadow-lg">
      {text}
    </div>
  );
}
