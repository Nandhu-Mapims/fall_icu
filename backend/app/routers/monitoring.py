# Purpose: API routes for health checks and frame analysis in ICU fall detection.
from fastapi import APIRouter, HTTPException

from app.schemas import AnalyzeRequest, DetectionResult, HealthResponse
from app.services.detector import FallDetectorService

router = APIRouter()
detector_service = FallDetectorService()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    data = detector_service.get_health()
    return HealthResponse(
        model_loaded=bool(data.get("model_loaded")),
        gpu_available=bool(data.get("gpu_available")),
        device=str(data.get("device") or "cpu"),
        model_path_requested=str(data.get("model_path_requested") or ""),
        model_loaded_from=data.get("model_loaded_from"),
        inference_image_size=int(data.get("inference_image_size") or 960),
        use_fp32_gpu_inference=bool(data.get("use_fp32_gpu_inference")),
        gpu_mem_total_mb=data.get("gpu_mem_total_mb"),
        gpu_mem_free_mb=data.get("gpu_mem_free_mb"),
        gpu_mem_reserved_mb=data.get("gpu_mem_reserved_mb"),
        tensorrt_engine=bool(data.get("tensorrt_engine")),
        inference_backend=_coerce_inference_backend(data.get("inference_backend")),
    )


def _coerce_inference_backend(value: object) -> str:
    if value in ("none", "tensorrt", "onnx", "pytorch"):
        return str(value)
    return "none"


@router.post("/analyze", response_model=DetectionResult)
def analyze_frame(payload: AnalyzeRequest) -> DetectionResult:
    try:
        return detector_service.analyze(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to process frame now.") from exc
