"use client";

import { useState } from "react";
import { exportDataset, ExportOut } from "@/lib/api";

const API = "http://localhost:8000";

export default function ExportPanel() {
  const [exporting, setExporting] = useState(false);
  const [result, setResult] = useState<ExportOut | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    setExporting(true);
    setError(null);
    setResult(null);

    try {
      const out = await exportDataset();
      setResult(out);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Export failed.");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div style={{ padding: "16px" }}>
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
        Export Dataset
      </h3>

      <button
        onClick={handleExport}
        disabled={exporting}
        style={{
          width: "100%",
          backgroundColor: exporting ? "#1c2e2e" : "#162424",
          color: exporting ? "#5a7d7d" : "#f2cc8f",
          border: "1px solid #f2cc8f44",
          borderRadius: 6,
          padding: "9px 0",
          fontSize: 14,
          fontWeight: 600,
          cursor: exporting ? "not-allowed" : "pointer",
          transition: "background-color 0.2s",
        }}
      >
        {exporting ? "Generating…" : "Export COCO JSON"}
      </button>

      {error && (
        <p style={{ fontSize: 12, color: "#e07a5f", marginTop: 10, marginBottom: 0 }}>
          {error}
        </p>
      )}

      {result && (
        <div
          style={{
            marginTop: 12,
            backgroundColor: "#1c2e2e",
            borderRadius: 8,
            padding: "12px 14px",
            border: "1px solid #81b29a44",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 13,
              color: "#8aacac",
              marginBottom: 10,
            }}
          >
            <span>{result.num_images} images</span>
            <span>{result.num_annotations} annotations</span>
          </div>

          <a
            href={`${API}${result.download_url}`}
            download={result.filename}
            style={{
              display: "block",
              textAlign: "center",
              backgroundColor: "#81b29a22",
              border: "1px solid #81b29a",
              color: "#81b29a",
              borderRadius: 6,
              padding: "7px 0",
              fontSize: 13,
              fontWeight: 600,
              textDecoration: "none",
            }}
          >
            Download {result.filename}
          </a>
        </div>
      )}
    </div>
  );
}
