"use client";

import { useState } from "react";
import { generateReport, ReportOut } from "@/lib/api";

interface Props {
  imageId: number | null;
  hasPredictions: boolean;
}

const RISK_COLORS: Record<string, string> = {
  High: "#e07a5f",
  Moderate: "#f2cc8f",
  Low: "#81b29a",
};

export default function ReportPanel({ imageId, hasPredictions }: Props) {
  const [report, setReport] = useState<ReportOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set());

  const handleGenerate = async () => {
    if (!imageId) return;
    setLoading(true);
    setError(null);
    setReport(null);
    setExpandedSections(new Set());

    try {
      const data = await generateReport(imageId);
      setReport(data);
      // Auto-expand the first 3 sections
      setExpandedSections(new Set([0, 1, 2]));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate report.");
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (idx: number) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  const handleDownload = () => {
    if (!report) return;

    const timestamp = new Date(report.generated_at).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });

    // Build a clean HTML document for printing/saving
    const sectionHtml = report.sections
      .map(
        (s) => `
        <div style="margin-bottom:24px;">
          <h2 style="font-size:16px;color:#2c3e50;border-bottom:2px solid #81b29a;
                     padding-bottom:6px;margin:0 0 10px;">${s.title}</h2>
          <pre style="font-family:'Georgia',serif;font-size:13px;line-height:1.7;
                      color:#333;white-space:pre-wrap;margin:0;">${s.content}</pre>
        </div>`
      )
      .join("");

    const riskColor = RISK_COLORS[report.risk_level] || "#5a7d7d";

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Clinical Report — ${report.image_name}</title>
  <style>
    body { font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 0 24px; color: #222; }
    .header { text-align: center; margin-bottom: 32px; padding-bottom: 20px; border-bottom: 3px solid #162424; }
    .header h1 { font-size: 24px; margin: 0 0 4px; color: #162424; }
    .header p { margin: 4px 0; font-size: 13px; color: #666; }
    .summary { display: flex; gap: 16px; margin-bottom: 28px; }
    .summary-card { flex: 1; background: #f8f9fa; border-radius: 8px; padding: 14px 16px; text-align: center; }
    .summary-card .value { font-size: 20px; font-weight: 700; margin: 0 0 2px; }
    .summary-card .label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    @media print { body { margin: 20px; } }
  </style>
</head>
<body>
  <div class="header">
    <h1>DermAnnotate Clinical Report</h1>
    <p>${report.image_name} &mdash; Generated ${timestamp}</p>
  </div>
  <div class="summary">
    <div class="summary-card">
      <p class="value">${report.primary_diagnosis}</p>
      <p class="label">Primary Diagnosis</p>
    </div>
    <div class="summary-card">
      <p class="value">${(report.confidence * 100).toFixed(1)}%</p>
      <p class="label">Confidence</p>
    </div>
    <div class="summary-card">
      <p class="value" style="color:${riskColor};">${report.risk_level}</p>
      <p class="label">Risk Level</p>
    </div>
  </div>
  ${sectionHtml}
</body>
</html>`;

    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${report.image_name.replace(/\.[^.]+$/, "")}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!imageId || !hasPredictions) {
    return (
      <div style={{ padding: "20px 16px", borderBottom: "1px solid #1c2e2e" }}>
        <p style={{ fontFamily: "Georgia, serif", fontSize: 13, color: "#5a7d7d", margin: 0 }}>
          {!imageId
            ? "Select an image to generate a report."
            : "Classify the image first to generate a clinical report."}
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
        Clinical Report
      </h3>

      {/* Generate button */}
      {!report && (
        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{
            width: "100%",
            padding: "9px 12px",
            fontSize: 13,
            fontFamily: "system-ui, sans-serif",
            fontWeight: 600,
            color: loading ? "#5a7d7d" : "#0f1a1a",
            backgroundColor: loading ? "#1c2e2e" : "#81b29a",
            border: "none",
            borderRadius: 6,
            cursor: loading ? "wait" : "pointer",
            transition: "background-color 0.15s",
          }}
        >
          {loading ? "Generating Report…" : "Generate Clinical Report"}
        </button>
      )}

      {error && (
        <p style={{ fontSize: 12, color: "#e07a5f", marginTop: 8 }}>{error}</p>
      )}

      {/* Report content */}
      {report && (
        <div style={{ marginTop: 4 }}>
          {/* Summary header */}
          <div
            style={{
              backgroundColor: "#1c2e2e",
              borderRadius: 8,
              padding: "14px",
              marginBottom: 10,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span style={{ fontSize: 15, fontWeight: 700, color: "#e8f0f0", fontFamily: "Georgia, serif" }}>
                {report.primary_diagnosis}
              </span>
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: RISK_COLORS[report.risk_level] || "#5a7d7d",
                  backgroundColor: `${RISK_COLORS[report.risk_level] || "#5a7d7d"}18`,
                  border: `1px solid ${RISK_COLORS[report.risk_level] || "#5a7d7d"}55`,
                  borderRadius: 4,
                  padding: "2px 8px",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                }}
              >
                {report.risk_level} Risk
              </span>
            </div>
            <div style={{ display: "flex", gap: 16, fontSize: 12, color: "#8aacac", fontFamily: "Consolas, monospace" }}>
              <span>Confidence: {(report.confidence * 100).toFixed(1)}%</span>
              <span>Model: {report.model_name}</span>
            </div>
          </div>

          {/* Collapsible sections */}
          {report.sections.map((section, idx) => {
            const isOpen = expandedSections.has(idx);
            return (
              <div key={idx} style={{ marginBottom: 4 }}>
                <button
                  onClick={() => toggleSection(idx)}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "8px 10px",
                    backgroundColor: isOpen ? "#1c2e2e" : "transparent",
                    border: "none",
                    borderRadius: isOpen ? "6px 6px 0 0" : 6,
                    cursor: "pointer",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    transition: "background-color 0.15s",
                  }}
                >
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: isOpen ? "#e8f0f0" : "#8aacac",
                      fontFamily: "system-ui, sans-serif",
                    }}
                  >
                    {section.title}
                  </span>
                  <span style={{ fontSize: 10, color: "#5a7d7d", transform: isOpen ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}>
                    ▼
                  </span>
                </button>
                {isOpen && (
                  <div
                    style={{
                      backgroundColor: "#1c2e2e",
                      borderRadius: "0 0 6px 6px",
                      padding: "8px 12px 12px",
                    }}
                  >
                    <pre
                      style={{
                        fontSize: 11.5,
                        lineHeight: 1.65,
                        color: "#c8d8d8",
                        fontFamily: "system-ui, sans-serif",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        margin: 0,
                      }}
                    >
                      {section.content}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}

          {/* Action buttons */}
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button
              onClick={handleDownload}
              style={{
                flex: 1,
                padding: "7px 12px",
                fontSize: 12,
                fontFamily: "system-ui, sans-serif",
                color: "#81b29a",
                backgroundColor: "#81b29a18",
                border: "1px solid #81b29a55",
                borderRadius: 5,
                cursor: "pointer",
                transition: "background-color 0.15s",
              }}
            >
              Download Report
            </button>
            <button
              onClick={handleGenerate}
              disabled={loading}
              style={{
                flex: 1,
                padding: "7px 12px",
                fontSize: 12,
                fontFamily: "system-ui, sans-serif",
                color: "#5a7d7d",
                backgroundColor: "transparent",
                border: "1px solid #2a4040",
                borderRadius: 5,
                cursor: loading ? "wait" : "pointer",
              }}
            >
              {loading ? "Regenerating…" : "Regenerate"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
