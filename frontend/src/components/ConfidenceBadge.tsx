"use client";

interface Props {
  confidence: number; // 0.0 – 1.0
  size?: "sm" | "md";
}

export default function ConfidenceBadge({ confidence, size = "md" }: Props) {
  const pct = Math.round(confidence * 100);
  const label = pct === 0 && confidence > 0 ? "<1%" : `${pct}%`;

  let color: string;
  if (confidence >= 0.7) {
    color = "#81b29a";
  } else if (confidence >= 0.4) {
    color = "#f2cc8f";
  } else {
    color = "#e07a5f";
  }

  const fontSize = size === "sm" ? "11px" : "13px";
  const padding = size === "sm" ? "2px 6px" : "3px 8px";

  return (
    <span
      style={{
        display: "inline-block",
        backgroundColor: `${color}22`,
        border: `1px solid ${color}`,
        color,
        borderRadius: 4,
        fontSize,
        fontFamily: "Consolas, monospace",
        fontWeight: 600,
        padding,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}
