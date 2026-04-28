"use client";

import { useState, useEffect, useCallback } from "react";
import {
  getAttentionMap,
  getAttentionInfo,
  attentionFullUrl,
  AttentionInfo,
} from "@/lib/api";

interface Props {
  imageId: number | null;
  hasPredictions: boolean;
  isMockMode: boolean;
}

export default function AttentionPanel({ imageId, hasPredictions, isMockMode }: Props) {
  const [info, setInfo] = useState<AttentionInfo | null>(null);
  const [layer, setLayer] = useState(11);
  const [head, setHead] = useState<number | null>(null); // null = average
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch model architecture info when image changes
  useEffect(() => {
    if (!imageId || !hasPredictions || isMockMode) {
      setInfo(null);
      setImageUrl(null);
      return;
    }
    getAttentionInfo(imageId)
      .then((data) => {
        setInfo(data);
        setLayer(Math.max(0, data.num_layers - 1));
      })
      .catch(() => setInfo(null));
  }, [imageId, hasPredictions, isMockMode]);

  const handleGenerate = useCallback(async () => {
    if (!imageId || !info) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getAttentionMap(imageId, layer, head);
      setImageUrl(attentionFullUrl(res.attention_url) + `?t=${Date.now()}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate attention map.");
    } finally {
      setLoading(false);
    }
  }, [imageId, info, layer, head]);

  if (!imageId || !hasPredictions || isMockMode) return null;
  if (!info || info.num_layers === 0) return null;

  return (
    <div style={{ padding: "16px", borderBottom: "1px solid #1c2e2e" }}>
      <h3
        style={{
          fontFamily: "Georgia, serif",
          fontSize: 13,
          color: "#8aacac",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          margin: "0 0 10px",
        }}
      >
        Attention Heads
      </h3>

      <p style={{ fontSize: 11, color: "#5a7d7d", margin: "0 0 10px", lineHeight: 1.5 }}>
        Visualize which image regions each transformer attention head focuses on.
        The CLS token attention pattern reveals learned diagnostic features.
      </p>

      {/* Layer selector */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: "#5a7d7d", fontFamily: "Consolas, monospace", minWidth: 40 }}>
          Layer
        </span>
        <input
          type="range"
          min={0}
          max={info.num_layers - 1}
          step={1}
          value={layer}
          onChange={(e) => setLayer(parseInt(e.target.value))}
          style={{ flex: 1, accentColor: "#81b29a" }}
        />
        <span style={{ fontSize: 11, color: "#8aacac", fontFamily: "Consolas, monospace", width: 28, textAlign: "right" }}>
          {layer}/{info.num_layers - 1}
        </span>
      </div>

      {/* Head selector */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 11, color: "#5a7d7d", fontFamily: "Consolas, monospace", minWidth: 40 }}>
          Head
        </span>
        <select
          value={head === null ? "avg" : head}
          onChange={(e) => setHead(e.target.value === "avg" ? null : parseInt(e.target.value))}
          style={{
            flex: 1,
            padding: "4px 8px",
            fontSize: 11,
            fontFamily: "Consolas, monospace",
            backgroundColor: "#1c2e2e",
            color: "#e8f0f0",
            border: "1px solid #2a4040",
            borderRadius: 4,
            cursor: "pointer",
          }}
        >
          <option value="avg">Average (all heads)</option>
          {Array.from({ length: info.num_heads }, (_, i) => (
            <option key={i} value={i}>
              Head {i}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={handleGenerate}
        disabled={loading}
        style={{
          width: "100%",
          padding: "7px 12px",
          fontSize: 12,
          fontFamily: "system-ui, sans-serif",
          color: loading ? "#5a7d7d" : "#f2cc8f",
          backgroundColor: loading ? "#1c2e2e" : "#f2cc8f18",
          border: "1px solid #f2cc8f55",
          borderRadius: 5,
          cursor: loading ? "wait" : "pointer",
          transition: "background-color 0.15s",
        }}
      >
        {loading ? "Generating…" : "Generate Attention Map"}
      </button>

      {error && (
        <p style={{ fontSize: 11, color: "#e07a5f", marginTop: 6, marginBottom: 0 }}>
          {error}
        </p>
      )}

      {/* Attention map display */}
      {imageUrl && (
        <div style={{ marginTop: 10 }}>
          <div
            style={{
              borderRadius: 6,
              overflow: "hidden",
              border: "1px solid #1c2e2e",
              backgroundColor: "#0f1a1a",
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl}
              alt={`Attention L${layer} H${head ?? "avg"}`}
              style={{
                width: "100%",
                display: "block",
              }}
              draggable={false}
            />
          </div>
          <p
            style={{
              fontSize: 10,
              color: "#5a7d7d",
              marginTop: 6,
              marginBottom: 0,
              fontFamily: "Consolas, monospace",
            }}
          >
            Layer {layer} · {head === null ? "Averaged" : `Head ${head}`} · {info.architecture}
          </p>

          {/* Interpretation guide */}
          <div
            style={{
              marginTop: 8,
              backgroundColor: "#1c2e2e",
              borderRadius: 6,
              padding: "8px 10px",
            }}
          >
            <div
              style={{
                height: 8,
                borderRadius: 2,
                background: "linear-gradient(90deg, #000004, #420a68, #932567, #dd513a, #fca50a, #fcffa4)",
                marginBottom: 4,
              }}
            />
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontSize: 9, color: "#5a7d7d" }}>Low attention</span>
              <span style={{ fontSize: 9, color: "#5a7d7d" }}>High attention</span>
            </div>
            <p style={{ fontSize: 10, color: "#5a7d7d", margin: "6px 0 0", lineHeight: 1.5 }}>
              Early layers (0-3) capture texture and edges. Middle layers (4-8) detect
              shapes and patterns. Late layers (9-11) focus on class-discriminative regions.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
