"use client";

import { useEffect, useState } from "react";
import { listImages, deleteImage, ImageListOut, ImageOut } from "@/lib/api";
import UploadZone from "./UploadZone";
import ConfidenceBadge from "./ConfidenceBadge";

interface Props {
  selectedImageId: number | null;
  onSelect: (id: number) => void;
  onRefresh: () => void;
  refreshTrigger: number;
}

const STATUS_DOT: Record<string, string> = {
  pending: "#5a7d7d",
  predicted: "#3d8282",
  annotated: "#f2cc8f",
  reviewed: "#81b29a",
};

export default function Sidebar({
  selectedImageId,
  onSelect,
  onRefresh,
  refreshTrigger,
}: Props) {
  const [images, setImages] = useState<ImageListOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  const fetchImages = async () => {
    setLoading(true);
    try {
      setImages(await listImages());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchImages();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshTrigger]);

  const handleDelete = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (!confirm("Delete this image and all its annotations?")) return;
    await deleteImage(id);
    setImages((prev) => prev.filter((img) => img.id !== id));
    if (selectedImageId === id) onSelect(-1);
  };

  const handleUploaded = async (image: ImageOut) => {
    await fetchImages();
    onSelect(image.id);
    onRefresh();
  };

  return (
    <div
      style={{
        width: 240,
        flexShrink: 0,
        backgroundColor: "#162424",
        borderRight: "1px solid #1c2e2e",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "16px 16px 10px",
          borderBottom: "1px solid #1c2e2e",
        }}
      >
        <h1
          style={{
            fontFamily: "Georgia, serif",
            fontSize: 16,
            color: "#e8f0f0",
            margin: "0 0 2px",
            letterSpacing: "0.02em",
          }}
        >
          DermAnnotate
        </h1>
        <p style={{ fontSize: 11, color: "#5a7d7d", margin: 0 }}>
          {images.length} image{images.length !== 1 ? "s" : ""}
        </p>
      </div>

      <UploadZone onUploaded={handleUploaded} />

      {/* Legend */}
      <div
        style={{
          display: "flex",
          gap: 10,
          padding: "8px 12px",
          flexWrap: "wrap",
        }}
      >
        {Object.entries(STATUS_DOT).map(([status, color]) => (
          <span
            key={status}
            style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: "#5a7d7d" }}
          >
            <span
              style={{ width: 7, height: 7, borderRadius: "50%", backgroundColor: color, flexShrink: 0 }}
            />
            {status}
          </span>
        ))}
      </div>

      {/* Image list */}
      <div style={{ flex: 1, overflowY: "auto", padding: "4px 0" }}>
        {loading && (
          <p style={{ textAlign: "center", color: "#5a7d7d", fontSize: 12, padding: 16 }}>
            Loading…
          </p>
        )}

        {!loading && images.length === 0 && (
          <p style={{ textAlign: "center", color: "#5a7d7d", fontSize: 12, padding: 16 }}>
            No images yet.
          </p>
        )}

        {images.map((img) => (
          <div
            key={img.id}
            onClick={() => onSelect(img.id)}
            onMouseEnter={() => setHoveredId(img.id)}
            onMouseLeave={() => setHoveredId(null)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "8px 12px",
              cursor: "pointer",
              backgroundColor:
                selectedImageId === img.id
                  ? "#1c2e2e"
                  : hoveredId === img.id
                  ? "#162e2e"
                  : "transparent",
              borderLeft: selectedImageId === img.id
                ? "2px solid #3d8282"
                : "2px solid transparent",
              transition: "background-color 0.15s",
            }}
          >
            {/* Thumbnail */}
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 4,
                overflow: "hidden",
                flexShrink: 0,
                backgroundColor: "#0f1a1a",
              }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`http://localhost:8000/uploads/${img.filename}`}
                alt={img.original_name}
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
                draggable={false}
              />
            </div>

            {/* Info */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 2 }}>
                <span
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: "50%",
                    backgroundColor: STATUS_DOT[img.status] ?? "#5a7d7d",
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontSize: 12,
                    color: "#e8f0f0",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {img.original_name}
                </span>
              </div>

              {img.top_label && img.top_confidence != null && (
                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span
                    style={{
                      fontSize: 10,
                      color: "#8aacac",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      flex: 1,
                    }}
                  >
                    {img.top_label}
                  </span>
                  <ConfidenceBadge confidence={img.top_confidence} size="sm" />
                </div>
              )}
            </div>

            {/* Delete */}
            <button
              onClick={(e) => handleDelete(e, img.id)}
              title="Delete image"
              style={{
                background: "none",
                border: "none",
                color: "#5a7d7d",
                cursor: "pointer",
                padding: "2px 4px",
                flexShrink: 0,
                fontSize: 14,
                lineHeight: 1,
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
