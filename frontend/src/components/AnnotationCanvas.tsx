"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { AnnotationOut, segmentImage } from "@/lib/api";

interface Point {
  x: number;
  y: number;
}

interface Props {
  width: number;
  height: number;
  existingAnnotations: AnnotationOut[];
  onPointsChange: (points: Point[]) => void;
  imageId: number | null;
  imageWidth: number;
  imageHeight: number;
}

type FabricModule = typeof import("fabric");

export default function AnnotationCanvas({
  width,
  height,
  existingAnnotations,
  onPointsChange,
  imageId,
  imageWidth,
  imageHeight,
}: Props) {
  const canvasElRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<{ canvas: InstanceType<FabricModule["Canvas"]>; mod: FabricModule } | null>(null);

  const [drawing, setDrawing] = useState(false);
  const [samMode, setSamMode] = useState(false);
  const [samLoading, setSamLoading] = useState(false);
  const pointsRef = useRef<Point[]>([]);
  // Fabric objects for in-progress polygon (dots + lines)
  const inProgressRef = useRef<object[]>([]);

  // Load Fabric and init canvas
  useEffect(() => {
    if (!canvasElRef.current || width === 0 || height === 0) return;

    let mounted = true;

    import("fabric").then((mod) => {
      if (!mounted || !canvasElRef.current) return;

      const canvas = new mod.Canvas(canvasElRef.current, {
        width,
        height,
        selection: true,
        backgroundColor: "transparent",
      });

      fabricRef.current = { canvas, mod };
      drawExisting(canvas, mod, existingAnnotations);
    });

    return () => {
      mounted = false;
      fabricRef.current?.canvas.dispose();
      fabricRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height]);

  // Re-render existing annotations when they change
  useEffect(() => {
    const ref = fabricRef.current;
    if (!ref) return;
    // Remove previous annotation overlays (keep in-progress objects)
    const objects = ref.canvas.getObjects();
    const toRemove = objects.filter((o) => (o as { _annotation?: boolean })._annotation);
    toRemove.forEach((o) => ref.canvas.remove(o));
    drawExisting(ref.canvas, ref.mod, existingAnnotations);
    ref.canvas.renderAll();
  }, [existingAnnotations]);

  function drawExisting(
    canvas: InstanceType<FabricModule["Canvas"]>,
    mod: FabricModule,
    annotations: AnnotationOut[]
  ) {
    annotations.forEach((ann) => {
      if (ann.annotation_type === "segmentation" && ann.geometry?.points?.length) {
        const poly = new mod.Polygon(ann.geometry.points, {
          fill: "rgba(61, 130, 130, 0.25)",
          stroke: "#3d8282",
          strokeWidth: 1.5,
          selectable: true,
          hasControls: false,
        });
        (poly as unknown as { _annotation: boolean })._annotation = true;
        canvas.add(poly);
      }
    });
    canvas.renderAll();
  }

  const handleCanvasClick = useCallback(
    async (e: React.MouseEvent<HTMLCanvasElement>) => {
      const ref = fabricRef.current;
      if (!ref) return;

      const rect = canvasElRef.current!.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // ── SAM mode: single click → auto-segment ──
      if (samMode && imageId != null) {
        setSamLoading(true);

        // Show pulsing dot at click position
        const dot = new ref.mod.Circle({
          left: x - 6,
          top: y - 6,
          radius: 6,
          fill: "#00b4d8",
          stroke: "#0096c7",
          strokeWidth: 2,
          selectable: false,
          opacity: 0.8,
        });
        (dot as unknown as { _sampulse: boolean })._sampulse = true;
        ref.canvas.add(dot);
        ref.canvas.renderAll();

        try {
          const result = await segmentImage(imageId, {
            x,
            y,
            display_width: width,
            display_height: height,
          });

          // Remove pulsing dot
          const toRemove = ref.canvas
            .getObjects()
            .filter((o) => (o as unknown as { _sampulse?: boolean })._sampulse);
          toRemove.forEach((o) => ref.canvas.remove(o));

          if (result.points.length >= 3) {
            const poly = new ref.mod.Polygon(result.points, {
              fill: "rgba(0, 180, 216, 0.25)",
              stroke: "#00b4d8",
              strokeWidth: 2,
              selectable: true,
              hasControls: false,
            });
            ref.canvas.add(poly);
            onPointsChange(result.points);
          }
        } catch (err) {
          // Remove pulsing dot on error
          const toRemove = ref.canvas
            .getObjects()
            .filter((o) => (o as unknown as { _sampulse?: boolean })._sampulse);
          toRemove.forEach((o) => ref.canvas.remove(o));
          console.error("SAM segmentation failed:", err);
        } finally {
          setSamLoading(false);
          setSamMode(false);
          ref.canvas.renderAll();
        }
        return;
      }

      // ── Manual drawing mode ──
      if (!drawing) return;

      const newPoint: Point = { x: Math.round(x), y: Math.round(y) };
      const newPoints = [...pointsRef.current, newPoint];
      pointsRef.current = newPoints;

      // Draw a dot at the clicked position
      const dot = new ref.mod.Circle({
        left: x - 4,
        top: y - 4,
        radius: 4,
        fill: "#e07a5f",
        selectable: false,
      });
      (dot as unknown as { _inprogress: boolean })._inprogress = true;
      ref.canvas.add(dot);

      // Draw line from previous point
      if (newPoints.length > 1) {
        const prev = newPoints[newPoints.length - 2];
        const line = new ref.mod.Line([prev.x, prev.y, x, y], {
          stroke: "#e07a5f",
          strokeWidth: 1.5,
          selectable: false,
          strokeDashArray: [4, 3],
        });
        (line as unknown as { _inprogress: boolean })._inprogress = true;
        ref.canvas.add(line);
      }

      ref.canvas.renderAll();
      onPointsChange(newPoints);
    },
    [drawing, samMode, imageId, width, height, onPointsChange]
  );

  const startDrawing = () => {
    setSamMode(false);
    setDrawing(true);
    pointsRef.current = [];
    onPointsChange([]);
  };

  const startSamMode = () => {
    setDrawing(false);
    pointsRef.current = [];
    setSamMode(true);
  };

  const finishPolygon = () => {
    const ref = fabricRef.current;
    if (!ref) return;

    const pts = pointsRef.current;

    // Remove in-progress dots/lines
    const toRemove = ref.canvas
      .getObjects()
      .filter((o) => (o as unknown as { _inprogress?: boolean })._inprogress);
    toRemove.forEach((o) => ref.canvas.remove(o));

    if (pts.length >= 3) {
      const poly = new ref.mod.Polygon(pts, {
        fill: "rgba(224, 122, 95, 0.25)",
        stroke: "#e07a5f",
        strokeWidth: 1.5,
        selectable: true,
        hasControls: false,
      });
      ref.canvas.add(poly);
    }

    ref.canvas.renderAll();
    setDrawing(false);
  };

  const clearCanvas = () => {
    const ref = fabricRef.current;
    if (!ref) return;

    // Remove everything except existing annotation overlays
    const toRemove = ref.canvas
      .getObjects()
      .filter((o) => !(o as unknown as { _annotation?: boolean })._annotation);
    toRemove.forEach((o) => ref.canvas.remove(o));

    pointsRef.current = [];
    inProgressRef.current = [];
    onPointsChange([]);
    setDrawing(false);
    setSamMode(false);
    ref.canvas.renderAll();
  };

  const deleteSelected = () => {
    const ref = fabricRef.current;
    if (!ref) return;
    const active = ref.canvas.getActiveObject();
    if (active) {
      ref.canvas.remove(active);
      ref.canvas.renderAll();
    }
  };

  const cursorStyle = samMode ? "crosshair" : drawing ? "crosshair" : "default";

  return (
    <div style={{ position: "relative", width, height }}>
      <canvas
        ref={canvasElRef}
        onClick={handleCanvasClick}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          cursor: cursorStyle,
        }}
      />

      {/* SAM loading overlay */}
      {samLoading && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width,
            height,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "rgba(0, 0, 0, 0.3)",
            borderRadius: 4,
            pointerEvents: "none",
            zIndex: 10,
          }}
        >
          <div
            style={{
              backgroundColor: "#0f1a1aee",
              padding: "8px 16px",
              borderRadius: 8,
              fontSize: 12,
              color: "#00b4d8",
              fontFamily: "system-ui, sans-serif",
              fontWeight: 600,
            }}
          >
            Segmenting...
          </div>
        </div>
      )}

      {/* Toolbar */}
      <div
        style={{
          position: "absolute",
          bottom: 10,
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          gap: 8,
          backgroundColor: "#0f1a1acc",
          padding: "6px 10px",
          borderRadius: 8,
          backdropFilter: "blur(4px)",
        }}
      >
        {!drawing && !samMode ? (
          <>
            <ToolBtn onClick={startSamMode} color="#00b4d8" disabled={imageId == null}>
              Auto-Segment
            </ToolBtn>
            <ToolBtn onClick={startDrawing} color="#e07a5f">
              Draw Polygon
            </ToolBtn>
            <ToolBtn onClick={deleteSelected} color="#5a7d7d">
              Delete Selected
            </ToolBtn>
            <ToolBtn onClick={clearCanvas} color="#5a7d7d">
              Clear All
            </ToolBtn>
          </>
        ) : samMode ? (
          <>
            <ToolBtn onClick={() => {}} color="#00b4d8" disabled>
              Click lesion to segment
            </ToolBtn>
            <ToolBtn onClick={() => setSamMode(false)} color="#5a7d7d">
              Cancel
            </ToolBtn>
          </>
        ) : (
          <>
            <ToolBtn onClick={finishPolygon} color="#81b29a">
              Finish ({pointsRef.current.length} pts)
            </ToolBtn>
            <ToolBtn onClick={clearCanvas} color="#5a7d7d">
              Cancel
            </ToolBtn>
          </>
        )}
      </div>
    </div>
  );
}

function ToolBtn({
  onClick,
  color,
  children,
  disabled,
}: {
  onClick: () => void;
  color: string;
  children: React.ReactNode;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        fontSize: 12,
        color: disabled ? "#5a7d7d" : color,
        backgroundColor: `${color}18`,
        border: `1px solid ${disabled ? "#2a4040" : color}55`,
        borderRadius: 5,
        padding: "4px 10px",
        cursor: disabled ? "not-allowed" : "pointer",
        fontFamily: "system-ui, sans-serif",
        whiteSpace: "nowrap",
        opacity: disabled ? 0.6 : 1,
      }}
    >
      {children}
    </button>
  );
}
