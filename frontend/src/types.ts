// Purpose: Shared frontend types for API and UI state.
export type PatientStatus = "HUMAN_DETECTED" | "FALL_DETECTED" | "NO_PERSON";

export interface Point {
  x: number;
  y: number;
}

export interface AnalyzeRequest {
  image_base64: string;
  patient_id: string;
  area_name: string;
}

export interface DetectionResult {
  patient_id: string;
  area_name: string;
  status: PatientStatus;
  confidence: number;
  used_cuda: boolean;
  message: string;
  bbox: Point | null;
  bbox_bottom_right: Point | null;
  person_count: number;
  people: PersonDetection[];
  device: string;
}

export interface ApiResult<T> {
  data: T | null;
  error: string | null;
}

export interface PersonDetection {
  person_id: number;
  status: PatientStatus;
  confidence: number;
  bbox: Point | null;
  bbox_bottom_right: Point | null;
  /** COCO order, normalized 0–1 (YOLO pose). */
  keypoints?: Point[];
}
