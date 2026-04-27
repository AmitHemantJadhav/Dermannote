"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ImageOut, AnnotationOut, imageUrl } from "@/lib/api";
import AnnotationCanvas from "./AnnotationCanvas";

interface Point {
  x: number;
  y: number;
}

interface Props {
  image: ImageOut | null;
  annotations: AnnotationOut[];
  onSegmentationChange: (points: Point[]) => void;
  gradcamUrl?: string | null;
  showGradCAM?: boolean;
  gradcamOpacity?: number;
}

const STATUS_COLORS: Record<string, string> = {
  pending:   "#5a7d7d",
  predicted: "#3d8282",
  annotated: "#f2cc8f",
  reviewed:  "#81b29a",
};

export default function ImageViewer({
  image,
  annotations,
  onSegmentationChange,
  gradcamUrl = null,
  showGradCAM = false,
  gradcamOpacity = 0.5,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });

  const compute = useCallback(() => {
    if (!image || !containerRef.current) return;
    const { offsetWidth: cw, offsetHeight: ch } = containerRef.current;
    if (!cw || !ch) return;

    const imgAspect = image.width / image.height;
    const containerAspect = cw / ch;

    let displayWidth: number;
    let displayHeight: number;

    if (imgAspect > containerAspect) {
      displayWidth = cw;
      displayHeight = cw / imgAspect;
    } else {
      displayHeight = ch;
      displayWidth = ch * imgAspect;
    }

    // Cap upscaling at 2.5× natural size so small images aren't blown up too large
    const maxUpscale = 2.5;
    displayWidth = Math.min(displayWidth, image.width * maxUpscale);
    displayHeight = Math.min(displayHeight, image.height * maxUpscale);

    setDisplaySize({
      width: Math.round(displayWidth),
      height: Math.round(displayHeight),
    });
  }, [image]);

  useEffect(() => {
    if (!image || !containerRef.current) return;

    compute();

    const ro = new ResizeObserver(compute);
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [image, compute]);

  if (!image) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          gap: 16,
        }}
      >
        <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#2a4040" strokeWidth="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <polyline points="21 15 16 10 5 21" />
        </svg>
        <div style={{ textAlign: "center" }}>
          <p style={{ fontFamily: "Georgia, serif", fontSize: 15, color: "#5a7d7d", margin: "0 0 6px" }}>
            No image selected
          </p>
          <p style={{ fontSize: 12, color: "#2a4040", margin: 0, fontFamily: "system-ui, sans-serif" }}>
            Upload or select an image from the sidebar
          </p>
        </div>
      </div>
    );
  }

  const statusColor = STATUS_COLORS[image.status] ?? "#5a7d7d";

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        backgroundColor: "#0a1414",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {/* Image — rendered at displaySize (capped at 2.5× natural, scaled down to fit) */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl(image.filename)}
        alt={image.original_name}
        style={{
          display: "block",
          width: displaySize.width > 0 ? displaySize.width : undefined,
          height: displaySize.height > 0 ? displaySize.height : undefined,
          maxWidth: "100%",
          maxHeight: "100%",
          objectFit: "contain",
          userSelect: "none",
          pointerEvents: "none",
        }}
        draggable={false}
      />

      {/* GradCAM heatmap overlay — between base image and annotation canvas */}
      {showGradCAM && gradcamUrl && displaySize.width > 0 && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            pointerEvents: "none",
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={gradcamUrl}
            alt="GradCAM heatmap"
            style={{
              display: "block",
              width: displaySize.width,
              height: displaySize.height,
              opacity: gradcamOpacity,
              objectFit: "contain",
              userSelect: "none",
            }}
            draggable={false}
          />
        </div>
      )}

      {/* Annotation canvas — centred absolutely over the image */}
      {displaySize.width > 0 && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            pointerEvents: "auto",
          }}
        >
          <AnnotationCanvas
            width={displaySize.width}
            height={displaySize.height}
            existingAnnotations={annotations}
            onPointsChange={onSegmentationChange}
            imageId={image.id}
            imageWidth={image.width}
            imageHeight={image.height}
          />
        </div>
      )}

      {/* GradCAM floating legend on image */}
      {showGradCAM && gradcamUrl && (
        <div
          style={{
            position: "absolute",
            bottom: 50,
            right: 16,
            backgroundColor: "#0f1a1add",
            borderRadius: 6,
            padding: "8px 12px",
            backdropFilter: "blur(6px)",
            pointerEvents: "none",
            minWidth: 140,
          }}
        >
          <p
            style={{
              fontSize: 10,
              fontWeight: 600,
              color: "#8aacac",
              margin: "0 0 4px",
              fontFamily: "system-ui, sans-serif",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Model Focus
          </p>
          <div
            style={{
              height: 8,
              borderRadius: 2,
              background: "linear-gradient(90deg, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)",
              marginBottom: 3,
            }}
          />
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontSize: 9, color: "#5a7d7d", fontFamily: "system-ui, sans-serif" }}>Low</span>
            <span style={{ fontSize: 9, color: "#5a7d7d", fontFamily: "system-ui, sans-serif" }}>High</span>
          </div>
        </div>
      )}

      {/* Image info bar */}
      <div
        style={{
          position: "absolute",
          top: 10,
          left: 10,
          backgroundColor: "#0f1a1acc",
          borderRadius: 6,
          padding: "5px 10px",
          fontSize: 11,
          fontFamily: "Consolas, monospace",
          color: "#8aacac",
          backdropFilter: "blur(4px)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span>{image.original_name}</span>
        <span style={{ color: "#2a4040" }}>·</span>
        <span>{image.width}×{image.height}px</span>
        <span style={{ color: "#2a4040" }}>·</span>
        <span
          style={{
            backgroundColor: `${statusColor}22`,
            color: statusColor,
            border: `1px solid ${statusColor}55`,
            borderRadius: 4,
            padding: "1px 7px",
            fontSize: 10,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
          }}
        >
          {image.status}
        </span>
      </div>
    </div>
  );
}
