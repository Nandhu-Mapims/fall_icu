# Purpose: Request and response schemas for pose-based ICU fall detection APIs.
from typing import Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float = Field(default=0.0, ge=0.0, le=1.0)
    y: float = Field(default=0.0, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    image_base64: str = Field(default="", min_length=1)
    patient_id: str = Field(default="unknown")
    area_name: str = Field(default="icu_bed_1")


class PersonDetection(BaseModel):
    person_id: int = 0
    status: Literal["HUMAN_DETECTED", "FALL_DETECTED", "NO_PERSON"] = "NO_PERSON"
    confidence: float = 0.0
    bbox: Point | None = None
    bbox_bottom_right: Point | None = None
    # COCO 17 keypoints, normalized 0–1 (same order as YOLO pose). Empty if unavailable.
    keypoints: list[Point] = Field(default_factory=list)


class DetectionResult(BaseModel):
    patient_id: str = "unknown"
    area_name: str = "icu_bed_1"
    status: Literal["HUMAN_DETECTED", "FALL_DETECTED", "NO_PERSON"] = "NO_PERSON"
    confidence: float = 0.0
    used_cuda: bool = False
    message: str = "No person detected."
    bbox: Point | None = None
    bbox_bottom_right: Point | None = None
    person_count: int = 0
    people: list[PersonDetection] = Field(default_factory=list)
    device: str = "cpu"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool = False
    gpu_available: bool = False
    device: str = "cpu"
    model_path_requested: str = ""
    model_loaded_from: str | None = None
    inference_image_size: int = 960
    use_fp32_gpu_inference: bool = False
    gpu_mem_total_mb: float | None = None
    gpu_mem_free_mb: float | None = None
    gpu_mem_reserved_mb: float | None = None
    tensorrt_engine: bool = False
    inference_backend: Literal["none", "tensorrt", "onnx", "pytorch"] = "none"
