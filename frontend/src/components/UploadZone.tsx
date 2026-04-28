"use client";

import { useState, useRef, DragEvent, ChangeEvent } from "react";
import { uploadImage, classifyImage, ImageOut } from "@/lib/api";

interface Props {
  onUploaded: (image: ImageOut) => void;
}

export default function UploadZone({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "uploading" | "classifying">("idle");
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFiles = async (files: File[]) => {
    const valid = files.filter((f) => ["image/jpeg", "image/png"].includes(f.type));
    if (valid.length === 0) {
      setError("Only JPEG and PNG files are accepted.");
      return;
    }

    setError(null);
    const isBatch = valid.length > 1;

    try {
      if (isBatch) {
        // Multi-file: upload all without classifying (use Batch Processing later)
        setStatus("uploading");
        setProgress({ done: 0, total: valid.length });
        let lastImage: ImageOut | null = null;
        for (let i = 0; i < valid.length; i++) {
          lastImage = await uploadImage(valid[i]);
          setProgress({ done: i + 1, total: valid.length });
        }
        if (lastImage) onUploaded(lastImage);
      } else {
        // Single file: upload and classify immediately
        setStatus("uploading");
        const image = await uploadImage(valid[0]);

        setStatus("classifying");
        await classifyImage(image.id);

        onUploaded(image);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setStatus("idle");
      setProgress({ done: 0, total: 0 });
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) processFiles(files);
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) processFiles(files);
    e.target.value = "";
  };

  const busy = status !== "idle";

  return (
    <div style={{ padding: "12px 12px 0" }}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !busy && inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragging ? "#3d8282" : "#2a4040"}`,
          borderRadius: 8,
          padding: "16px 12px",
          textAlign: "center",
          cursor: busy ? "not-allowed" : "pointer",
          transition: "border-color 0.2s, background-color 0.2s",
          backgroundColor: dragging ? "#3d828210" : "transparent",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          multiple
          onChange={onFileChange}
          style={{ display: "none" }}
        />

        {busy ? (
          <div>
            <Spinner />
            <p style={{ fontSize: 12, color: "#8aacac", margin: "8px 0 0" }}>
              {status === "uploading" && progress.total > 1
                ? `Uploading ${progress.done}/${progress.total}…`
                : status === "uploading"
                  ? "Uploading…"
                  : "Classifying…"}
            </p>
          </div>
        ) : (
          <>
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#5a7d7d"
              strokeWidth="1.5"
              style={{ display: "block", margin: "0 auto 6px" }}
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p style={{ fontSize: 12, color: "#5a7d7d", margin: 0 }}>
              Drop images or click
            </p>
          </>
        )}
      </div>

      {error && (
        <p style={{ fontSize: 11, color: "#e07a5f", margin: "6px 0 0", textAlign: "center" }}>
          {error}
        </p>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <div
      style={{
        width: 20,
        height: 20,
        border: "2px solid #2a4040",
        borderTop: "2px solid #3d8282",
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
        margin: "0 auto",
      }}
    >
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
