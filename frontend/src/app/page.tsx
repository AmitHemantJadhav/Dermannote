"use client";

import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import ImageViewer from "@/components/ImageViewer";
import PredictionPanel from "@/components/PredictionPanel";
import AttentionPanel from "@/components/AttentionPanel";
import AnnotationForm from "@/components/AnnotationForm";
import ExportPanel from "@/components/ExportPanel";
import ReportPanel from "@/components/ReportPanel";
import BatchPanel from "@/components/BatchPanel";
import ReviewQueuePanel from "@/components/ReviewQueuePanel";
import EmbeddingViz from "@/components/EmbeddingViz";
import {
  getImage,
  deleteAnnotation,
  getGradCAM,
  gradcamFullUrl,
  getQueueStats,
  ImageDetailOut,
  PredictionOut,
  AnnotationOut,
} from "@/lib/api";

interface Point {
  x: number;
  y: number;
}

type TabKey = "analysis" | "queue" | "explore";

export default function Home() {
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImageDetailOut | null>(null);
  const [predictions, setPredictions] = useState<PredictionOut[]>([]);
  const [annotations, setAnnotations] = useState<AnnotationOut[]>([]);
  const [acceptedLabel, setAcceptedLabel] = useState("");
  const [canvasPoints, setCanvasPoints] = useState<Point[]>([]);
  const [sidebarRefresh, setSidebarRefresh] = useState(0);
  const [loadingImage, setLoadingImage] = useState(false);
  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);
  const [showGradCAM, setShowGradCAM] = useState(false);
  const [gradcamOpacity, setGradcamOpacity] = useState(0.5);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>("analysis");
  const [pendingCount, setPendingCount] = useState(0);

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
    setGradcamUrl(null);
    setShowGradCAM(false);
    setGradcamLoading(false);

    try {
      const detail = await getImage(id);
      setSelectedImage(detail);
      setPredictions(detail.predictions);
      setAnnotations(detail.annotations);
    } finally {
      setLoadingImage(false);
    }
  }, []);

  // Refresh pending count for queue badge
  useEffect(() => {
    getQueueStats()
      .then((s) => setPendingCount(s.pending))
      .catch(() => {});
  }, [sidebarRefresh]);

  const handleAnnotationSaved = async (annotation: AnnotationOut) => {
    setAnnotations((prev) => [annotation, ...prev]);
    setCanvasPoints([]);
    setSidebarRefresh((n) => n + 1);
  };

  const isMockMode = predictions.length > 0 && predictions[0]?.model_name === "mock-v1";

  const handleToggleGradCAM = useCallback(async () => {
    if (showGradCAM) {
      setShowGradCAM(false);
      return;
    }

    if (!selectedImageId) return;

    if (gradcamUrl) {
      setShowGradCAM(true);
      return;
    }

    setGradcamLoading(true);
    try {
      const res = await getGradCAM(selectedImageId);
      setGradcamUrl(gradcamFullUrl(res.gradcam_url) + `?t=${Date.now()}`);
      setShowGradCAM(true);
    } catch (err) {
      console.error("GradCAM generation failed:", err);
    } finally {
      setGradcamLoading(false);
    }
  }, [showGradCAM, selectedImageId, gradcamUrl]);

  const handleDeleteAnnotation = async (id: number) => {
    await deleteAnnotation(id);
    setAnnotations((prev) => prev.filter((a) => a.id !== id));
  };

  const tabs: { key: TabKey; label: string }[] = [
    { key: "analysis", label: "Analysis" },
    { key: "queue", label: "Review Queue" },
    { key: "explore", label: "Explore" },
  ];

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
            gradcamUrl={gradcamUrl}
            showGradCAM={showGradCAM}
            gradcamOpacity={gradcamOpacity}
          />
        )}
      </main>

      {/* Right: tabbed aside */}
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
        {/* Tab bar */}
        <div
          style={{
            display: "flex",
            borderBottom: "1px solid #1c2e2e",
            flexShrink: 0,
          }}
        >
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                flex: 1,
                padding: "10px 0",
                fontSize: 11,
                fontFamily: "Georgia, serif",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                color: activeTab === tab.key ? "#e8f0f0" : "#5a7d7d",
                backgroundColor: "transparent",
                border: "none",
                borderBottom:
                  activeTab === tab.key
                    ? "2px solid #81b29a"
                    : "2px solid transparent",
                cursor: "pointer",
                transition: "color 0.15s, border-color 0.15s",
                position: "relative",
              }}
            >
              {tab.label}
              {tab.key === "queue" && pendingCount > 0 && (
                <span
                  style={{
                    marginLeft: 6,
                    fontSize: 10,
                    fontFamily: "Consolas, monospace",
                    backgroundColor: "#e07a5f",
                    color: "#0f1a1a",
                    borderRadius: 8,
                    padding: "1px 6px",
                    fontWeight: 700,
                  }}
                >
                  {pendingCount}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === "analysis" && (
          <>
            <PredictionPanel
              predictions={predictions}
              onAccept={setAcceptedLabel}
              showGradCAM={showGradCAM}
              gradcamLoading={gradcamLoading}
              gradcamOpacity={gradcamOpacity}
              onToggleGradCAM={handleToggleGradCAM}
              onGradCAMOpacityChange={setGradcamOpacity}
              isMockMode={isMockMode}
            />

            <AttentionPanel
              imageId={selectedImageId}
              hasPredictions={predictions.length > 0}
              isMockMode={isMockMode}
            />

            <ReportPanel
              imageId={selectedImageId}
              hasPredictions={predictions.length > 0}
            />

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
          </>
        )}

        {activeTab === "queue" && (
          <ReviewQueuePanel
            selectedImageId={selectedImageId}
            onSelectImage={loadImageData}
            refreshTrigger={sidebarRefresh}
          />
        )}

        {activeTab === "explore" && (
          <>
            <BatchPanel onComplete={() => setSidebarRefresh((n) => n + 1)} />
            <EmbeddingViz
              onSelectImage={(id) => {
                loadImageData(id);
                setActiveTab("analysis");
              }}
              selectedImageId={selectedImageId}
            />
          </>
        )}
      </aside>
    </div>
  );
}
