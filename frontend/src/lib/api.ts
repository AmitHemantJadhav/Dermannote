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

export interface GradCAMResponse {
  gradcam_url: string;
}

export async function getGradCAM(imageId: number): Promise<GradCAMResponse> {
  return apiFetch<GradCAMResponse>(`/api/predictions/${imageId}/gradcam`, {
    method: "POST",
  });
}

export function gradcamFullUrl(path: string): string {
  return `${API}${path}`;
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

// ── Active Learning Queue ──

export interface QueueItemOut {
  id: number;
  image_id: number;
  max_confidence: number;
  margin: number;
  entropy: number;
  uncertainty_score: number;
  review_status: string;
  reviewer?: string;
  corrected_label?: string;
  review_notes?: string;
  reviewed_at?: string;
  created_at: string;
  filename?: string;
  original_name?: string;
  image_status?: string;
  top_label?: string;
  top_confidence?: number;
}

export interface ReviewCreate {
  status: string;
  corrected_label?: string;
  review_notes?: string;
  reviewer?: string;
}

export interface QueueStatsOut {
  total: number;
  pending: number;
  confirmed: number;
  corrected: number;
  skipped: number;
  avg_uncertainty: number;
}

export async function getQueue(status?: string): Promise<QueueItemOut[]> {
  const qs = status ? `?status=${status}` : "";
  return apiFetch<QueueItemOut[]>(`/api/queue/${qs}`);
}

export async function getQueueStats(): Promise<QueueStatsOut> {
  return apiFetch<QueueStatsOut>("/api/queue/stats");
}

export async function reviewImage(
  imageId: number,
  data: ReviewCreate
): Promise<QueueItemOut> {
  return apiFetch<QueueItemOut>(`/api/queue/${imageId}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

// ── SAM Auto-Segmentation ──

export interface SegmentationRequest {
  x: number;
  y: number;
  display_width: number;
  display_height: number;
}

export interface SegmentationResponse {
  points: { x: number; y: number }[];
}

export async function segmentImage(
  imageId: number,
  data: SegmentationRequest
): Promise<SegmentationResponse> {
  return apiFetch<SegmentationResponse>(`/api/segmentation/${imageId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

// ── Batch Processing ──

export interface BatchItemResult {
  image_id: number;
  filename: string;
  success: boolean;
  top_label?: string;
  top_confidence?: number;
  error?: string;
}

export interface BatchOut {
  total: number;
  succeeded: number;
  failed: number;
  results: BatchItemResult[];
}

export async function batchClassify(imageIds?: number[]): Promise<BatchOut> {
  return apiFetch<BatchOut>("/api/batch/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_ids: imageIds ?? null }),
  });
}

// ── Embedding Visualization ──

export interface EmbeddingPoint {
  image_id: number;
  x: number;
  y: number;
  label: string;
  filename: string;
}

export interface TSNEOut {
  num_images: number;
  perplexity: number;
  points: EmbeddingPoint[];
}

export async function computeTSNE(
  perplexity?: number,
  imageIds?: number[],
): Promise<TSNEOut> {
  return apiFetch<TSNEOut>("/api/embeddings/tsne", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      perplexity: perplexity ?? 30,
      image_ids: imageIds ?? null,
    }),
  });
}

// ── Attention Head Visualization ──

export interface AttentionResponse {
  attention_url: string;
  layer: number;
  head: number | null;
}

export interface AttentionInfo {
  num_layers: number;
  num_heads: number;
  architecture: string;
}

export async function getAttentionMap(
  imageId: number,
  layer: number,
  head: number | null,
): Promise<AttentionResponse> {
  return apiFetch<AttentionResponse>(`/api/attention/${imageId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layer, head }),
  });
}

export function attentionFullUrl(path: string): string {
  return `${API}${path}`;
}

export async function getAttentionInfo(imageId: number): Promise<AttentionInfo> {
  return apiFetch<AttentionInfo>(`/api/attention/${imageId}/info`);
}

// ── Clinical Report ──

export interface ReportSection {
  title: string;
  content: string;
}

export interface ReportOut {
  image_id: number;
  image_name: string;
  generated_at: string;
  primary_diagnosis: string;
  confidence: number;
  risk_level: string;
  model_name: string;
  sections: ReportSection[];
}

export async function generateReport(imageId: number): Promise<ReportOut> {
  return apiFetch<ReportOut>(`/api/reports/${imageId}`, { method: "POST" });
}

// ── Export ──

export async function exportDataset(imageIds?: number[]): Promise<ExportOut> {
  return apiFetch<ExportOut>("/api/export/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format: "coco", image_ids: imageIds ?? null }),
  });
}
