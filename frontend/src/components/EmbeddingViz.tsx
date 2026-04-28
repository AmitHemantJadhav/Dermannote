"use client";

import { useState, useMemo } from "react";
import { computeTSNE, EmbeddingPoint } from "@/lib/api";

interface Props {
  onSelectImage: (id: number) => void;
  selectedImageId: number | null;
}

// Color palette for the 7 HAM10000 classes
const LABEL_COLORS: Record<string, string> = {
  "Melanocytic Nevi":      "#81b29a",
  "Melanoma":              "#e07a5f",
  "Benign Keratosis":      "#f2cc8f",
  "Basal Cell Carcinoma":  "#3d8282",
  "Actinic Keratosis":     "#c98bb9",
  "Vascular Lesion":       "#7eb5d6",
  "Dermatofibroma":        "#d4a574",
};

const PLOT_SIZE = 280;
const PLOT_PADDING = 28;

export default function EmbeddingViz({ onSelectImage, selectedImageId }: Props) {
  const [points, setPoints] = useState<EmbeddingPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [perplexity, setPerplexity] = useState(30);

  const handleCompute = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await computeTSNE(perplexity);
      setPoints(res.points);
    } catch (err) {
      setError(err instanceof Error ? err.message : "t-SNE computation failed.");
    } finally {
      setLoading(false);
    }
  };

  // Unique labels present in the current result
  const uniqueLabels = useMemo(() => {
    const labels = new Set(points.map((p) => p.label));
    return Array.from(labels).sort();
  }, [points]);

  const toSvgX = (x: number) => PLOT_PADDING + ((x + 1) / 2) * (PLOT_SIZE - 2 * PLOT_PADDING);
  const toSvgY = (y: number) => PLOT_PADDING + ((y + 1) / 2) * (PLOT_SIZE - 2 * PLOT_PADDING);

  const hoveredPoint = points.find((p) => p.image_id === hoveredId);

  return (
    <div style={{ padding: "16px" }}>
      <h3
        style={{
          fontFamily: "Georgia, serif",
          fontSize: 13,
          color: "#8aacac",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          margin: "0 0 12px",
        }}
      >
        Embedding Space (t-SNE)
      </h3>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 10, alignItems: "center" }}>
        <button
          onClick={handleCompute}
          disabled={loading}
          style={{
            flex: 1,
            padding: "7px 12px",
            fontSize: 12,
            fontFamily: "system-ui, sans-serif",
            fontWeight: 600,
            color: loading ? "#5a7d7d" : "#0f1a1a",
            backgroundColor: loading ? "#1c2e2e" : "#81b29a",
            border: "none",
            borderRadius: 5,
            cursor: loading ? "wait" : "pointer",
          }}
        >
          {loading ? "Computing…" : points.length > 0 ? "Recompute" : "Compute t-SNE"}
        </button>
      </div>

      {/* Perplexity control */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 11, color: "#5a7d7d", fontFamily: "Consolas, monospace" }}>
          Perplexity
        </span>
        <input
          type="range"
          min={5}
          max={50}
          step={5}
          value={perplexity}
          onChange={(e) => setPerplexity(parseInt(e.target.value))}
          style={{ flex: 1, accentColor: "#81b29a" }}
        />
        <span style={{ fontSize: 11, color: "#8aacac", fontFamily: "Consolas, monospace", width: 24, textAlign: "right" }}>
          {perplexity}
        </span>
      </div>

      {error && (
        <p style={{ fontSize: 11, color: "#e07a5f", margin: "0 0 8px" }}>{error}</p>
      )}

      {/* Scatter plot */}
      {points.length > 0 && (
        <>
          <div
            style={{
              backgroundColor: "#0f1a1a",
              borderRadius: 8,
              border: "1px solid #1c2e2e",
              overflow: "hidden",
              position: "relative",
            }}
          >
            <svg
              width={PLOT_SIZE}
              height={PLOT_SIZE}
              viewBox={`0 0 ${PLOT_SIZE} ${PLOT_SIZE}`}
              style={{ display: "block" }}
            >
              {/* Grid lines */}
              <line x1={PLOT_PADDING} y1={PLOT_SIZE / 2} x2={PLOT_SIZE - PLOT_PADDING} y2={PLOT_SIZE / 2} stroke="#1c2e2e" strokeWidth={0.5} />
              <line x1={PLOT_SIZE / 2} y1={PLOT_PADDING} x2={PLOT_SIZE / 2} y2={PLOT_SIZE - PLOT_PADDING} stroke="#1c2e2e" strokeWidth={0.5} />

              {/* Data points */}
              {points.map((p) => {
                const isSelected = p.image_id === selectedImageId;
                const isHovered = p.image_id === hoveredId;
                const color = LABEL_COLORS[p.label] ?? "#5a7d7d";
                const r = isSelected ? 6 : isHovered ? 5 : 4;

                return (
                  <circle
                    key={p.image_id}
                    cx={toSvgX(p.x)}
                    cy={toSvgY(p.y)}
                    r={r}
                    fill={color}
                    fillOpacity={isSelected || isHovered ? 1.0 : 0.7}
                    stroke={isSelected ? "#e8f0f0" : "none"}
                    strokeWidth={isSelected ? 2 : 0}
                    style={{ cursor: "pointer", transition: "r 0.1s" }}
                    onMouseEnter={() => setHoveredId(p.image_id)}
                    onMouseLeave={() => setHoveredId(null)}
                    onClick={() => onSelectImage(p.image_id)}
                  />
                );
              })}
            </svg>

            {/* Hover tooltip */}
            {hoveredPoint && (
              <div
                style={{
                  position: "absolute",
                  top: 6,
                  left: 6,
                  backgroundColor: "#162424ee",
                  borderRadius: 5,
                  padding: "5px 8px",
                  fontSize: 11,
                  color: "#e8f0f0",
                  pointerEvents: "none",
                  maxWidth: 200,
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 2 }}>
                  {hoveredPoint.filename}
                </div>
                <div style={{ color: LABEL_COLORS[hoveredPoint.label] ?? "#8aacac" }}>
                  {hoveredPoint.label}
                </div>
              </div>
            )}
          </div>

          {/* Legend */}
          <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6 }}>
            {uniqueLabels.map((label) => (
              <span
                key={label}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  fontSize: 10,
                  color: "#8aacac",
                }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    backgroundColor: LABEL_COLORS[label] ?? "#5a7d7d",
                    flexShrink: 0,
                  }}
                />
                {label}
              </span>
            ))}
          </div>

          <p style={{ fontSize: 10, color: "#5a7d7d", margin: "8px 0 0", fontFamily: "Consolas, monospace" }}>
            {points.length} images projected
          </p>
        </>
      )}

      {points.length === 0 && !loading && !error && (
        <p style={{ fontSize: 12, color: "#5a7d7d", margin: 0 }}>
          Classify images first, then compute t-SNE to see how the model organizes them in feature space.
        </p>
      )}
    </div>
  );
}
