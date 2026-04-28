"use client";

import { useState } from "react";
import { batchClassify, BatchOut } from "@/lib/api";

interface Props {
  onComplete: () => void;
}

export default function BatchPanel({ onComplete }: Props) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BatchOut | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleBatch = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await batchClassify();
      setResult(res);
      if (res.succeeded > 0) onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch classification failed.");
    } finally {
      setLoading(false);
    }
  };

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
        Batch Processing
      </h3>

      <button
        onClick={handleBatch}
        disabled={loading}
        style={{
          width: "100%",
          padding: "8px 12px",
          fontSize: 12,
          fontFamily: "system-ui, sans-serif",
          fontWeight: 600,
          color: loading ? "#5a7d7d" : "#e8f0f0",
          backgroundColor: loading ? "#1c2e2e" : "#3d828244",
          border: "1px solid #3d828266",
          borderRadius: 5,
          cursor: loading ? "wait" : "pointer",
          transition: "background-color 0.15s",
        }}
      >
        {loading ? "Classifying…" : "Classify All Pending Images"}
      </button>

      {error && (
        <p style={{ fontSize: 11, color: "#e07a5f", marginTop: 6, marginBottom: 0 }}>
          {error}
        </p>
      )}

      {result && (
        <div style={{ marginTop: 10 }}>
          {result.total === 0 ? (
            <p style={{ fontSize: 12, color: "#5a7d7d", margin: 0 }}>
              No pending images to classify.
            </p>
          ) : (
            <>
              {/* Summary */}
              <div
                style={{
                  display: "flex",
                  gap: 8,
                  marginBottom: 8,
                }}
              >
                <div
                  style={{
                    flex: 1,
                    backgroundColor: "#1c2e2e",
                    borderRadius: 6,
                    padding: "8px 10px",
                    textAlign: "center",
                  }}
                >
                  <p
                    style={{
                      fontSize: 18,
                      fontWeight: 700,
                      color: "#81b29a",
                      margin: "0 0 2px",
                      fontFamily: "Consolas, monospace",
                    }}
                  >
                    {result.succeeded}
                  </p>
                  <p style={{ fontSize: 10, color: "#5a7d7d", margin: 0, textTransform: "uppercase" }}>
                    Classified
                  </p>
                </div>
                {result.failed > 0 && (
                  <div
                    style={{
                      flex: 1,
                      backgroundColor: "#1c2e2e",
                      borderRadius: 6,
                      padding: "8px 10px",
                      textAlign: "center",
                    }}
                  >
                    <p
                      style={{
                        fontSize: 18,
                        fontWeight: 700,
                        color: "#e07a5f",
                        margin: "0 0 2px",
                        fontFamily: "Consolas, monospace",
                      }}
                    >
                      {result.failed}
                    </p>
                    <p style={{ fontSize: 10, color: "#5a7d7d", margin: 0, textTransform: "uppercase" }}>
                      Failed
                    </p>
                  </div>
                )}
              </div>

              {/* Per-image results */}
              <div style={{ maxHeight: 180, overflowY: "auto" }}>
                {result.results.map((r) => (
                  <div
                    key={r.image_id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "4px 0",
                      borderBottom: "1px solid #1c2e2e",
                      fontSize: 11,
                    }}
                  >
                    <span
                      style={{
                        color: "#c8d8d8",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        maxWidth: 130,
                      }}
                    >
                      {r.filename}
                    </span>
                    {r.success ? (
                      <span style={{ color: "#81b29a", fontFamily: "Consolas, monospace" }}>
                        {r.top_label ? `${r.top_label} ${Math.round((r.top_confidence ?? 0) * 100)}%` : "OK"}
                      </span>
                    ) : (
                      <span style={{ color: "#e07a5f" }}>Error</span>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
