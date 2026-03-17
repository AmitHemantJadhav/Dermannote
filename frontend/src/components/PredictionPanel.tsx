"use client";

import { useState, useEffect } from "react";
import { PredictionOut } from "@/lib/api";
import ConfidenceBadge from "./ConfidenceBadge";

interface Props {
  predictions: PredictionOut[];
  onAccept: (label: string) => void;
}

const DISEASE_INFO: Record<string, string> = {
  "Melanocytic Nevi":      "Common benign mole — monitor for ABCDE changes",
  "Melanoma":              "Malignant skin cancer — requires urgent evaluation",
  "Benign Keratosis":      "Non-cancerous seborrheic or actinic keratosis",
  "Basal Cell Carcinoma":  "Most common skin cancer — slow-growing, rarely spreads",
  "Actinic Keratosis":     "UV-induced pre-cancerous lesion — may progress to SCC",
  "Vascular Lesion":       "Blood vessel abnormality — typically benign",
  "Dermatofibroma":        "Benign fibrous nodule — firm, usually harmless",
};

function confidenceColor(c: number): string {
  if (c >= 0.7) return "#81b29a";
  if (c >= 0.4) return "#f2cc8f";
  return "#e07a5f";
}

export default function PredictionPanel({ predictions, onAccept }: Props) {
  const [accepted, setAccepted] = useState<string>("");
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  // Reset when a new image is loaded
  useEffect(() => {
    setAccepted("");
  }, [predictions]);

  const handleAccept = (label: string) => {
    setAccepted(label);
    onAccept(label);
  };

  if (predictions.length === 0) {
    return (
      <div style={{ padding: "20px 16px", borderBottom: "1px solid #1c2e2e" }}>
        <p style={{ fontFamily: "Georgia, serif", fontSize: 13, color: "#5a7d7d", margin: 0 }}>
          No predictions yet. Upload and classify an image.
        </p>
      </div>
    );
  }

  return (
    <div style={{ padding: "16px", borderBottom: "1px solid #1c2e2e" }}>
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
        ML Predictions
      </h3>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {predictions.map((pred) => {
          const color = confidenceColor(pred.confidence);
          const pct = Math.round(pred.confidence * 100);
          const isAccepted = accepted === pred.label;
          const isHovered = hoveredId === pred.id;
          const desc = DISEASE_INFO[pred.label];

          return (
            <div
              key={pred.id}
              style={{
                backgroundColor: isAccepted ? `${color}14` : "#1c2e2e",
                borderRadius: 8,
                padding: "12px 14px",
                border: isAccepted
                  ? `1px solid ${color}77`
                  : pred.rank === 1
                  ? `1px solid ${color}44`
                  : "1px solid transparent",
                transition: "background-color 0.2s, border-color 0.2s",
              }}
            >
              {/* Header row */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  marginBottom: 4,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span
                    style={{
                      width: 20,
                      height: 20,
                      borderRadius: "50%",
                      backgroundColor: isAccepted ? `${color}22` : "#162424",
                      border: `1px solid ${color}`,
                      color,
                      fontSize: 11,
                      fontFamily: "Consolas, monospace",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                      transition: "background-color 0.2s",
                    }}
                  >
                    {isAccepted ? "✓" : pred.rank}
                  </span>
                  <span style={{ fontWeight: 600, fontSize: 14, color: "#e8f0f0", lineHeight: 1.2 }}>
                    {pred.label}
                  </span>
                </div>
                <ConfidenceBadge confidence={pred.confidence} size="sm" />
              </div>

              {/* Disease description */}
              {desc && (
                <p
                  style={{
                    fontSize: 11,
                    color: "#5a7d7d",
                    margin: "0 0 8px",
                    paddingLeft: 28,
                    lineHeight: 1.5,
                    fontFamily: "system-ui, sans-serif",
                  }}
                >
                  {desc}
                </p>
              )}

              {/* Confidence bar */}
              <div
                style={{
                  height: 4,
                  borderRadius: 2,
                  backgroundColor: "#0f1a1a",
                  marginBottom: 10,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${pct}%`,
                    background: `linear-gradient(90deg, ${color}88, ${color})`,
                    borderRadius: 2,
                    transition: "width 0.4s ease",
                  }}
                />
              </div>

              {/* Accept / Accepted */}
              {isAccepted ? (
                <div
                  style={{
                    fontSize: 12,
                    color,
                    fontFamily: "system-ui, sans-serif",
                    display: "flex",
                    alignItems: "center",
                    gap: 5,
                  }}
                >
                  <span>✓</span>
                  <span>Accepted — label copied to form</span>
                </div>
              ) : (
                <button
                  onClick={() => handleAccept(pred.label)}
                  onMouseEnter={() => setHoveredId(pred.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  style={{
                    fontSize: 12,
                    color: isHovered ? "#e8f0f0" : color,
                    backgroundColor: isHovered ? `${color}40` : `${color}18`,
                    border: `1px solid ${color}55`,
                    borderRadius: 4,
                    padding: "4px 12px",
                    cursor: "pointer",
                    fontFamily: "system-ui, sans-serif",
                    transition: "background-color 0.15s, color 0.15s",
                  }}
                >
                  Accept
                </button>
              )}
            </div>
          );
        })}
      </div>

      <p
        style={{
          fontSize: 11,
          color: "#5a7d7d",
          marginTop: 10,
          marginBottom: 0,
          fontFamily: "Consolas, monospace",
        }}
      >
        model: {predictions[0]?.model_name}
      </p>
    </div>
  );
}
