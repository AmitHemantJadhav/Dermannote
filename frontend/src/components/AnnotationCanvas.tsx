"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { AnnotationOut } from "@/lib/api";

interface Point {
  x: number;
  y: number;
}

interface Props {
  width: number;
  height: number;
  existingAnnotations: AnnotationOut[];
  onPointsChange: (points: Point[]) => void;
}

type FabricModule = typeof import("fabric");

export default function AnnotationCanvas({
  width,
  height,
  existingAnnotations,
  onPointsChange,
}: Props) {
  const canvasElRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<{ canvas: InstanceType<FabricModule["Canvas"]>; mod: FabricModule } | null>(null);

  const [drawing, setDrawing] = useState(false);
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
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!drawing || !fabricRef.current) return;

      const ref = fabricRef.current;
      const rect = canvasElRef.current!.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

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
    [drawing, onPointsChange]
  );

  const startDrawing = () => {
    setDrawing(true);
    pointsRef.current = [];
    onPointsChange([]);
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

  return (
    <div style={{ position: "relative", width, height }}>
      <canvas
        ref={canvasElRef}
        onClick={handleCanvasClick}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          cursor: drawing ? "crosshair" : "default",
        }}
      />

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
        {!drawing ? (
          <ToolBtn onClick={startDrawing} color="#e07a5f">
            Draw Polygon
          </ToolBtn>
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
        {!drawing && (
          <>
            <ToolBtn onClick={deleteSelected} color="#5a7d7d">
              Delete Selected
            </ToolBtn>
            <ToolBtn onClick={clearCanvas} color="#5a7d7d">
              Clear All
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
}: {
  onClick: () => void;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        fontSize: 12,
        color,
        backgroundColor: `${color}18`,
        border: `1px solid ${color}55`,
        borderRadius: 5,
        padding: "4px 10px",
        cursor: "pointer",
        fontFamily: "system-ui, sans-serif",
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </button>
  );
}
