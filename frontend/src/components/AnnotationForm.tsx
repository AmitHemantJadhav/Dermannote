"use client";

import { useState, useEffect } from "react";
import { createAnnotation, AnnotationOut } from "@/lib/api";

interface Point {
  x: number;
  y: number;
}

interface Props {
  imageId: number | null;
  initialLabel: string;
  segmentationPoints: Point[];
  onSaved: (annotation: AnnotationOut) => void;
}

export default function AnnotationForm({
  imageId,
  initialLabel,
  segmentationPoints,
  onSaved,
}: Props) {
  const [label, setLabel] = useState("");
  const [annotationType, setAnnotationType] = useState<"classification" | "segmentation">(
    "classification"
  );
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveBtnHovered, setSaveBtnHovered] = useState(false);

  // Pre-fill label when user clicks "Accept" on a prediction
  useEffect(() => {
    if (initialLabel) setLabel(initialLabel);
  }, [initialLabel]);

  const handleSave = async () => {
    if (!imageId) return;
    if (!label.trim()) {
      setError("Diagnosis label is required.");
      return;
    }
    if (annotationType === "segmentation" && segmentationPoints.length < 3) {
      setError("Draw at least 3 points on the canvas for segmentation.");
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const annotation = await createAnnotation(imageId, {
        label: label.trim(),
        annotation_type: annotationType,
        geometry:
          annotationType === "segmentation" && segmentationPoints.length >= 3
            ? { points: segmentationPoints }
            : null,
        notes: notes.trim() || undefined,
      });
      onSaved(annotation);
      setNotes("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to save annotation.");
    } finally {
      setSaving(false);
    }
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    backgroundColor: "#0f1a1a",
    border: "1px solid #2a4040",
    borderRadius: 6,
    color: "#e8f0f0",
    fontSize: 13,
    padding: "8px 10px",
    outline: "none",
    fontFamily: "system-ui, sans-serif",
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    fontSize: 11,
    color: "#8aacac",
    textTransform: "uppercase",
    letterSpacing: "0.07em",
    marginBottom: 5,
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
          margin: "0 0 14px",
        }}
      >
        Annotation
      </h3>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div>
          <label style={labelStyle}>Diagnosis</label>
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="e.g. Melanoma"
            style={inputStyle}
            disabled={!imageId}
          />
        </div>

        <div>
          <label style={labelStyle}>Type</label>
          <select
            value={annotationType}
            onChange={(e) =>
              setAnnotationType(e.target.value as "classification" | "segmentation")
            }
            style={{ ...inputStyle, cursor: "pointer" }}
            disabled={!imageId}
          >
            <option value="classification">Classification</option>
            <option value="segmentation">Segmentation</option>
          </select>
        </div>

        {annotationType === "segmentation" && (
          <p style={{ fontSize: 12, color: "#5a7d7d", margin: 0 }}>
            {segmentationPoints.length} point(s) drawn.
            {segmentationPoints.length > 0 && segmentationPoints.length < 3 && (
              <span style={{ color: "#e07a5f" }}> Need at least 3.</span>
            )}
            {segmentationPoints.length >= 3 && (
              <span style={{ color: "#81b29a" }}> Ready to save.</span>
            )}
          </p>
        )}

        <div>
          <label style={labelStyle}>Clinical Notes</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Optional clinical observations..."
            rows={3}
            style={{ ...inputStyle, resize: "vertical" }}
            disabled={!imageId}
          />
        </div>

        {error && (
          <p style={{ fontSize: 12, color: "#e07a5f", margin: 0 }}>{error}</p>
        )}

        <button
          onClick={handleSave}
          disabled={!imageId || saving}
          onMouseEnter={() => setSaveBtnHovered(true)}
          onMouseLeave={() => setSaveBtnHovered(false)}
          style={{
            backgroundColor: !imageId
              ? "#1c2e2e"
              : saveBtnHovered
              ? "#4a9494"
              : "#3d8282",
            color: imageId ? "#e8f0f0" : "#5a7d7d",
            border: "none",
            borderRadius: 6,
            padding: "9px 0",
            fontSize: 14,
            fontWeight: 600,
            cursor: imageId ? "pointer" : "not-allowed",
            transition: "background-color 0.2s",
          }}
        >
          {saving ? "Saving…" : "Save Annotation"}
        </button>
      </div>
    </div>
  );
}
