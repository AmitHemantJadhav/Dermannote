"use client";

import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import ImageViewer from "@/components/ImageViewer";
import PredictionPanel from "@/components/PredictionPanel";
import AnnotationForm from "@/components/AnnotationForm";
import ExportPanel from "@/components/ExportPanel";
import {
  getImage,
  deleteAnnotation,
  ImageDetailOut,
  PredictionOut,
  AnnotationOut,
} from "@/lib/api";

interface Point {
  x: number;
  y: number;
}

export default function Home() {
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImageDetailOut | null>(null);
  const [predictions, setPredictions] = useState<PredictionOut[]>([]);
  const [annotations, setAnnotations] = useState<AnnotationOut[]>([]);
  const [acceptedLabel, setAcceptedLabel] = useState("");
  const [canvasPoints, setCanvasPoints] = useState<Point[]>([]);
  const [sidebarRefresh, setSidebarRefresh] = useState(0);
  const [loadingImage, setLoadingImage] = useState(false);

  const loadImageData = useCallback(async (id: number) => {
    if (id < 0) {
      setSelectedImageId(null);
      setSelectedImage(null);
      setPredictions([]);
      setAnnotations([]);
      return;
    }

    setLoadingImage(true);
    setSelectedImageId(id);
    setAcceptedLabel("");
    setCanvasPoints([]);

    try {
      const detail = await getImage(id);
      setSelectedImage(detail);
      setPredictions(detail.predictions);
      setAnnotations(detail.annotations);
    } finally {
      setLoadingImage(false);
    }
  }, []);

  const handleAnnotationSaved = async (annotation: AnnotationOut) => {
    setAnnotations((prev) => [annotation, ...prev]);
    setCanvasPoints([]);
    // Refresh sidebar so status badge updates
    setSidebarRefresh((n) => n + 1);
  };

  const handleDeleteAnnotation = async (id: number) => {
    await deleteAnnotation(id);
    setAnnotations((prev) => prev.filter((a) => a.id !== id));
  };

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Left: sidebar */}
      <Sidebar
        selectedImageId={selectedImageId}
        onSelect={loadImageData}
        onRefresh={() => setSidebarRefresh((n) => n + 1)}
        refreshTrigger={sidebarRefresh}
      />

      {/* Center: image viewer */}
      <main
        style={{
          flex: 1,
          overflow: "hidden",
          position: "relative",
          backgroundColor: "#0a1414",
        }}
      >
        {loadingImage ? (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: "#5a7d7d",
              fontSize: 14,
              fontFamily: "Georgia, serif",
            }}
          >
            Loading…
          </div>
        ) : (
          <ImageViewer
            image={selectedImage}
            annotations={annotations}
            onSegmentationChange={setCanvasPoints}
          />
        )}
      </main>

      {/* Right: predictions + annotation form + export */}
      <aside
        style={{
          width: 320,
          flexShrink: 0,
          backgroundColor: "#162424",
          borderLeft: "1px solid #1c2e2e",
          display: "flex",
          flexDirection: "column",
          overflowY: "auto",
        }}
      >
        <PredictionPanel predictions={predictions} onAccept={setAcceptedLabel} />

        <AnnotationForm
          imageId={selectedImageId}
          initialLabel={acceptedLabel}
          segmentationPoints={canvasPoints}
          onSaved={handleAnnotationSaved}
        />

        {/* Annotation history */}
        {annotations.length > 0 && (
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
              Saved Annotations
            </h3>
            {annotations.map((ann) => (
              <div
                key={ann.id}
                style={{
                  backgroundColor: "#1c2e2e",
                  borderRadius: 6,
                  padding: "8px 10px",
                  marginBottom: 6,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: 8,
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: "#e8f0f0", margin: "0 0 2px" }}>
                    {ann.label}
                  </p>
                  <p style={{ fontSize: 11, color: "#5a7d7d", margin: 0 }}>
                    {ann.annotation_type}
                    {ann.notes && ` · ${ann.notes.slice(0, 40)}…`}
                  </p>
                </div>
                <button
                  onClick={() => handleDeleteAnnotation(ann.id)}
                  title="Delete annotation"
                  style={{
                    background: "none",
                    border: "none",
                    color: "#5a7d7d",
                    cursor: "pointer",
                    fontSize: 16,
                    padding: "0 2px",
                    flexShrink: 0,
                    lineHeight: 1,
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <ExportPanel />
      </aside>
    </div>
  );
}
