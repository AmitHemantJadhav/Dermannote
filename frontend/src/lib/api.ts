const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface ImageOut {
  id: number;
  filename: string;
  original_name: string;
  width: number;
  height: number;
  uploaded_at: string;
  status: string;
}

export interface ImageListOut {
  id: number;
  filename: string;
  original_name: string;
  status: string;
  top_label?: string;
  top_confidence?: number;
  uploaded_at: string;
}

export interface PredictionOut {
  id: number;
  image_id: number;
  model_name: string;
  label: string;
  confidence: number;
  rank: number;
  created_at: string;
}

export interface AnnotationCreate {
  label: string;
  annotation_type?: string;
  geometry?: { points: { x: number; y: number }[] } | null;
  notes?: string;
  annotator?: string;
}

export interface AnnotationOut {
  id: number;
  image_id: number;
  annotator: string;
  label: string;
  annotation_type: string;
  geometry?: { points: { x: number; y: number }[] } | null;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface ImageDetailOut extends ImageOut {
  predictions: PredictionOut[];
  annotations: AnnotationOut[];
}

export interface ExportOut {
  download_url: string;
  num_images: number;
  num_annotations: number;
  filename: string;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export function imageUrl(filename: string): string {
  return `${API}/uploads/${filename}`;
}

export async function uploadImage(file: File): Promise<ImageOut> {
  const form = new FormData();
  form.append("file", file);
  return apiFetch<ImageOut>("/api/images/upload", { method: "POST", body: form });
}

export async function listImages(status?: string): Promise<ImageListOut[]> {
  const qs = status ? `?status=${status}` : "";
  return apiFetch<ImageListOut[]>(`/api/images/${qs}`);
}

export async function getImage(id: number): Promise<ImageDetailOut> {
  return apiFetch<ImageDetailOut>(`/api/images/${id}`);
}

export async function deleteImage(id: number): Promise<void> {
  await apiFetch<unknown>(`/api/images/${id}`, { method: "DELETE" });
}

export async function classifyImage(imageId: number): Promise<PredictionOut[]> {
  return apiFetch<PredictionOut[]>(`/api/predictions/${imageId}/classify`, {
    method: "POST",
  });
}

export async function getPredictions(imageId: number): Promise<PredictionOut[]> {
  return apiFetch<PredictionOut[]>(`/api/predictions/${imageId}`);
}

export async function createAnnotation(
  imageId: number,
  data: AnnotationCreate
): Promise<AnnotationOut> {
  return apiFetch<AnnotationOut>(`/api/annotations/${imageId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function getAnnotations(imageId: number): Promise<AnnotationOut[]> {
  return apiFetch<AnnotationOut[]>(`/api/annotations/${imageId}`);
}

export async function deleteAnnotation(annotationId: number): Promise<void> {
  await apiFetch<unknown>(`/api/annotations/${annotationId}`, {
    method: "DELETE",
  });
}

export async function exportDataset(imageIds?: number[]): Promise<ExportOut> {
  return apiFetch<ExportOut>("/api/export/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format: "coco", image_ids: imageIds ?? null }),
  });
}
