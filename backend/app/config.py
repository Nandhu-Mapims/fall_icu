# Purpose: Centralized runtime configuration for the FastAPI fall detection service.
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "ICU Fall Detection API"
    api_prefix: str = "/api/v1"
    # TensorRT .engine from YOLO("yolo26n-pose.pt").export(format="engine", ...). Imgsz must match inference_image_size.
    model_path: str = str(Path("models/yolo26x-pose.engine"))
    # Ultralytics predict `conf`: lower = more person boxes (noisier); used with pose head.
    confidence_threshold: float = 0.25
    # Minimum per-keypoint confidence to treat a joint as usable in fall / lay-down heuristics (0–1).
    keypoint_confidence_min: float = 0.2
    # Larger imgsz uses more VRAM and usually improves pose quality (try 960–1280 on ~6GB).
    inference_image_size: int = 960
    max_people_per_frame: int = 8
    # FP16 on GPU: fast, lower VRAM. Set use_fp32_gpu_inference=true to roughly double VRAM use.
    use_half_precision: bool = True
    use_fp32_gpu_inference: bool = False
    cuda_device: int = 0
    model_warmup_runs: int = 2
    request_timeout_seconds: int = 20
    require_cuda: bool = True

    # When true and MODEL_PATH is .engine but file missing, run export (needs working TensorRT). Default off to avoid error-35 loops.
    tensorrt_auto_export_if_missing: bool = False
    # If TensorRT export fails (e.g. CUDA error 35), still load PyTorch weights so the API works.
    tensorrt_allow_pytorch_fallback: bool = True
    # If an ONNX file exists from a previous failed TensorRT step, skip re-export unless true (avoids slow boot loops).
    tensorrt_retry_export_after_onnx: bool = False

settings = Settings()
