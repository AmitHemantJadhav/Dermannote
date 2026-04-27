"use client";

import { useState, useEffect, useCallback } from "react";
import {
  getQueue,
  getQueueStats,
  reviewImage,
  imageUrl,
  QueueItemOut,
  QueueStatsOut,
} from "@/lib/api";

const LABELS = [
  "Melanocytic Nevi",
  "Melanoma",
  "Benign Keratosis",
  "Basal Cell Carcinoma",
  "Actinic Keratosis",
  "Vascular Lesion",
  "Dermatofibroma",
];

function uncertaintyColor(score: number): string {
  if (score >= 0.6) return "#e07a5f";
  if (score >= 0.35) return "#f2cc8f";
  return "#81b29a";
}

function statusBadge(status: string): { label: string; color: string } {
  switch (status) {
    case "confirmed":
      return { label: "Confirmed", color: "#81b29a" };
    case "corrected":
      return { label: "Corrected", color: "#f2cc8f" };
    case "skipped":
      return { label: "Skipped", color: "#5a7d7d" };
    default:
      return { label: "Pending", color: "#e07a5f" };
  }
}

interface Props {
  selectedImageId: number | null;
  onSelectImage: (id: number) => void;
  refreshTrigger?: number;
}

export default function ReviewQueuePanel({
  selectedImageId,
  onSelectImage,
  refreshTrigger = 0,
}: Props) {
  const [items, setItems] = useState<QueueItemOut[]>([]);
  const [stats, setStats] = useState<QueueStatsOut | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [loading, setLoading] = useState(false);
  const [correctingId, setCorrectingId] = useState<number | null>(null);
  const [correctedLabel, setCorrectedLabel] = useState("");
  const [reviewNotes, setReviewNotes] = useState("");

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [q, s] = await Promise.all([
        getQueue(filter === "all" ? undefined : filter),
        getQueueStats(),
      ]);
      setItems(q);
      setStats(s);
    } catch (err) {
      console.error("Failed to load queue:", err);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    fetchData();
  }, [fetchData, refreshTrigger]);

  const handleReview = async (
    imageId: number,
    status: "confirmed" | "corrected" | "skipped"
  ) => {
    try {
      await reviewImage(imageId, {
        status,
        corrected_label: status === "corrected" ? correctedLabel : undefined,
        review_notes: reviewNotes || undefined,
      });
      setCorrectingId(null);
      setCorrectedLabel("");
      setReviewNotes("");
      fetchData();
    } catch (err) {
      console.error("Review failed:", err);
    }
  };

  return (
    <div style={{ padding: "16px", display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Stats header */}
      {stats && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: 6,
          }}
        >
          {[
            { label: "Pending", value: stats.pending, color: "#e07a5f" },
            { label: "Reviewed", value: stats.confirmed + stats.corrected, color: "#81b29a" },
            { label: "Avg Uncert.", value: `${Math.round(stats.avg_uncertainty * 100)}%`, color: "#f2cc8f" },
          ].map((s) => (
            <div
              key={s.label}
              style={{
                backgroundColor: "#1c2e2e",
                borderRadius: 6,
                padding: "8px 10px",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 700,
                  color: s.color,
                  fontFamily: "Consolas, monospace",
                }}
              >
                {s.value}
              </div>
              <div
                style={{
                  fontSize: 10,
                  color: "#5a7d7d",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginTop: 2,
                }}
              >
                {s.label}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Filter dropdown */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <label
          style={{
            fontSize: 11,
            color: "#5a7d7d",
            fontFamily: "Consolas, monospace",
            flexShrink: 0,
          }}
        >
          Filter:
        </label>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{
            flex: 1,
            backgroundColor: "#1c2e2e",
            color: "#e8f0f0",
            border: "1px solid #2a4040",
            borderRadius: 4,
            padding: "4px 8px",
            fontSize: 12,
            fontFamily: "system-ui, sans-serif",
          }}
        >
          <option value="all">All</option>
          <option value="pending">Pending</option>
          <option value="confirmed">Confirmed</option>
          <option value="corrected">Corrected</option>
          <option value="skipped">Skipped</option>
        </select>
      </div>

      {/* Queue list */}
      {loading ? (
        <div style={{ color: "#5a7d7d", fontSize: 13, textAlign: "center", padding: 20 }}>
          Loading queue...
        </div>
      ) : items.length === 0 ? (
        <div style={{ color: "#5a7d7d", fontSize: 13, textAlign: "center", padding: 20 }}>
          No items in queue. Classify images to populate.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {items.map((item) => {
            const uColor = uncertaintyColor(item.uncertainty_score);
            const badge = statusBadge(item.review_status);
            const isSelected = selectedImageId === item.image_id;
            const isCorrectingThis = correctingId === item.image_id;

            return (
              <div key={item.id}>
                <div
                  onClick={() => onSelectImage(item.image_id)}
                  style={{
                    backgroundColor: isSelected ? "#1c3838" : "#1c2e2e",
                    borderRadius: 8,
                    padding: "10px 12px",
                    cursor: "pointer",
                    border: isSelected
                      ? "1px solid #3d5a5a"
                      : "1px solid transparent",
                    transition: "background-color 0.15s, border-color 0.15s",
                  }}
                >
                  {/* Top row: thumbnail + info */}
                  <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                    {item.filename && (
                      <img
                        src={imageUrl(item.filename)}
                        alt=""
                        style={{
                          width: 40,
                          height: 40,
                          borderRadius: 4,
                          objectFit: "cover",
                          flexShrink: 0,
                        }}
                      />
                    )}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: 13,
                          fontWeight: 600,
                          color: "#e8f0f0",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {item.original_name || `Image #${item.image_id}`}
                      </div>
                      <div style={{ fontSize: 11, color: "#5a7d7d", marginTop: 2 }}>
                        {item.top_label || "No prediction"}
                        {item.top_confidence != null &&
                          ` · ${Math.round(item.top_confidence * 100)}%`}
                      </div>
                    </div>
                    {/* Status badge */}
                    <span
                      style={{
                        fontSize: 10,
                        color: badge.color,
                        border: `1px solid ${badge.color}55`,
                        borderRadius: 4,
                        padding: "2px 6px",
                        flexShrink: 0,
                        fontFamily: "Consolas, monospace",
                      }}
                    >
                      {badge.label}
                    </span>
                  </div>

                  {/* Uncertainty bar */}
                  <div style={{ marginTop: 8 }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginBottom: 3,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 10,
                          color: "#5a7d7d",
                          fontFamily: "Consolas, monospace",
                        }}
                      >
                        uncertainty
                      </span>
                      <span
                        style={{
                          fontSize: 10,
                          color: uColor,
                          fontFamily: "Consolas, monospace",
                          fontWeight: 600,
                        }}
                      >
                        {Math.round(item.uncertainty_score * 100)}%
                      </span>
                    </div>
                    <div
                      style={{
                        height: 4,
                        borderRadius: 2,
                        backgroundColor: "#0f1a1a",
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${Math.round(item.uncertainty_score * 100)}%`,
                          background: `linear-gradient(90deg, ${uColor}88, ${uColor})`,
                          borderRadius: 2,
                          transition: "width 0.3s ease",
                        }}
                      />
                    </div>
                  </div>

                  {/* Review actions — only show for selected pending items */}
                  {isSelected && item.review_status === "pending" && (
                    <div
                      style={{
                        marginTop: 10,
                        display: "flex",
                        gap: 6,
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        onClick={() => handleReview(item.image_id, "confirmed")}
                        style={{
                          flex: 1,
                          padding: "5px 0",
                          fontSize: 11,
                          fontFamily: "system-ui, sans-serif",
                          color: "#81b29a",
                          backgroundColor: "#81b29a18",
                          border: "1px solid #81b29a55",
                          borderRadius: 4,
                          cursor: "pointer",
                        }}
                      >
                        Confirm
                      </button>
                      <button
                        onClick={() => {
                          setCorrectingId(isCorrectingThis ? null : item.image_id);
                          setCorrectedLabel("");
                        }}
                        style={{
                          flex: 1,
                          padding: "5px 0",
                          fontSize: 11,
                          fontFamily: "system-ui, sans-serif",
                          color: "#f2cc8f",
                          backgroundColor: isCorrectingThis ? "#f2cc8f22" : "#f2cc8f18",
                          border: "1px solid #f2cc8f55",
                          borderRadius: 4,
                          cursor: "pointer",
                        }}
                      >
                        Correct
                      </button>
                      <button
                        onClick={() => handleReview(item.image_id, "skipped")}
                        style={{
                          flex: 1,
                          padding: "5px 0",
                          fontSize: 11,
                          fontFamily: "system-ui, sans-serif",
                          color: "#5a7d7d",
                          backgroundColor: "#5a7d7d18",
                          border: "1px solid #5a7d7d55",
                          borderRadius: 4,
                          cursor: "pointer",
                        }}
                      >
                        Skip
                      </button>
                    </div>
                  )}
                </div>

                {/* Correction form — shown below the card */}
                {isCorrectingThis && (
                  <div
                    style={{
                      backgroundColor: "#162424",
                      border: "1px solid #f2cc8f44",
                      borderRadius: "0 0 8px 8px",
                      padding: "10px 12px",
                      marginTop: -2,
                      display: "flex",
                      flexDirection: "column",
                      gap: 8,
                    }}
                  >
                    <select
                      value={correctedLabel}
                      onChange={(e) => setCorrectedLabel(e.target.value)}
                      style={{
                        backgroundColor: "#1c2e2e",
                        color: "#e8f0f0",
                        border: "1px solid #2a4040",
                        borderRadius: 4,
                        padding: "5px 8px",
                        fontSize: 12,
                      }}
                    >
                      <option value="">Select correct label...</option>
                      {LABELS.map((l) => (
                        <option key={l} value={l}>
                          {l}
                        </option>
                      ))}
                    </select>
                    <input
                      type="text"
                      placeholder="Notes (optional)"
                      value={reviewNotes}
                      onChange={(e) => setReviewNotes(e.target.value)}
                      style={{
                        backgroundColor: "#1c2e2e",
                        color: "#e8f0f0",
                        border: "1px solid #2a4040",
                        borderRadius: 4,
                        padding: "5px 8px",
                        fontSize: 12,
                      }}
                    />
                    <button
                      disabled={!correctedLabel}
                      onClick={() => handleReview(item.image_id, "corrected")}
                      style={{
                        padding: "5px 12px",
                        fontSize: 11,
                        color: correctedLabel ? "#f2cc8f" : "#5a7d7d",
                        backgroundColor: "#f2cc8f18",
                        border: "1px solid #f2cc8f55",
                        borderRadius: 4,
                        cursor: correctedLabel ? "pointer" : "not-allowed",
                        alignSelf: "flex-end",
                      }}
                    >
                      Submit Correction
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
