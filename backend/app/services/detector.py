# Purpose: YOLO pose model inference service with CUDA fallback and ICU-specific fall heuristics.
from __future__ import annotations

import base64
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import settings
from app.schemas import AnalyzeRequest, DetectionResult, PersonDetection, Point
logger = logging.getLogger(__name__)

NOSE_INDEX = 0
LEFT_EYE_INDEX = 1
RIGHT_EYE_INDEX = 2
LEFT_EAR_INDEX = 3
RIGHT_EAR_INDEX = 4
LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6
LEFT_ELBOW_INDEX = 7
RIGHT_ELBOW_INDEX = 8
LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10
LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
LEFT_KNEE_INDEX = 13
RIGHT_KNEE_INDEX = 14
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16

HEAD_POINT_INDICES: tuple[int, ...] = (
    NOSE_INDEX,
    LEFT_EYE_INDEX,
    RIGHT_EYE_INDEX,
    LEFT_EAR_INDEX,
    RIGHT_EAR_INDEX,
)

# Fall / lay-down heuristics (normalized image coords; y grows downward).
TORSO_HORIZONTAL_MAX_DY = 0.18
HEAD_TO_HIP_VERTICAL_SPAN_MAX = 0.22
HIP_LOW_Y = 0.55
LEG_HORIZONTAL_MAX_DY = 0.14
BBOX_WIDE_RATIO_MIN = 1.12
# Head decoupled from trunk (lying with head up or chin down); y increases downward.
HEAD_CLEAR_ABOVE_SHOULDER_MIN = 0.035
HEAD_DROPPED_BELOW_SHOULDER_MIN = 0.05
ANKLE_FLOOR_Y = 0.64
PARTIAL_LEG_HIP_KNEE_MAX_DY = 0.16
SHOULDER_SPREAD_MIN_X = 0.1
SHOULDER_LINE_MAX_DY = 0.12
ARM_CHAIN_MAX_DY = 0.15
LEG_SINGLE_SIDE_SCALE = 1.38
WRIST_LOW_Y = 0.58
WRIST_PAIR_MAX_DY = 0.14
# Close-up / partial body: model often guesses hips near shoulders → false "horizontal torso" + compact span + wide face box.
# Require real torso corners + at least one knee or ankle before prone/fall rules (except full bilateral leg chain).
# Knee/ankle must sit near or below shoulder band in image y (reject facial keypoints mis-labeled as legs).
ANCHOR_MAX_Y_ABOVE_SHOULDER = 0.08

STATUS_PRIORITY = {
    "NO_PERSON": 0,
    "HUMAN_DETECTED": 1,
    "FALL_DETECTED": 2,
}


def _head_reference_y(
    xy: np.ndarray,
    joint_usable: Callable[[int], bool],
    nose_y: float,
) -> tuple[float, int]:
    """Average y of usable head keypoints (nose, eyes, ears); count how many contributed."""
    ys: list[float] = []
    for idx in HEAD_POINT_INDICES:
        if joint_usable(idx):
            ys.append(float(xy[idx][1]))
    if not ys:
        return nose_y, 0
    return sum(ys) / len(ys), len(ys)


def _single_leg_chain_horizontal(
    xy: np.ndarray,
    joint_usable: Callable[[int], bool],
    hip_i: int,
    knee_i: int,
    ankle_i: int,
) -> bool:
    if not (joint_usable(hip_i) and joint_usable(knee_i) and joint_usable(ankle_i)):
        return False
    ys = [float(xy[hip_i][1]), float(xy[knee_i][1]), float(xy[ankle_i][1])]
    return (max(ys) - min(ys)) < LEG_HORIZONTAL_MAX_DY * LEG_SINGLE_SIDE_SCALE


def _arm_chain_horizontal(
    xy: np.ndarray,
    joint_usable: Callable[[int], bool],
    shoulder_i: int,
    elbow_i: int,
    wrist_i: int,
) -> bool:
    if not (joint_usable(shoulder_i) and joint_usable(elbow_i) and joint_usable(wrist_i)):
        return False
    ys = [float(xy[shoulder_i][1]), float(xy[elbow_i][1]), float(xy[wrist_i][1])]
    return (max(ys) - min(ys)) < ARM_CHAIN_MAX_DY


def _wrists_low_and_level(
    xy: np.ndarray,
    joint_usable: Callable[[int], bool],
    wrist_low_y: float,
    max_dy: float,
) -> bool:
    if not (joint_usable(LEFT_WRIST_INDEX) and joint_usable(RIGHT_WRIST_INDEX)):
        return False
    ly = float(xy[LEFT_WRIST_INDEX][1])
    ry = float(xy[RIGHT_WRIST_INDEX][1])
    if ly < wrist_low_y or ry < wrist_low_y:
        return False
    return abs(ly - ry) < max_dy


def _extra_prone_cues(
    *,
    lower_body_anchor_ok: bool,
    horizontal_torso: bool,
    hips_low: bool,
    nose_ok: bool,
    head_ref_y: float,
    head_point_count: int,
    shoulder_mid_y: float,
    center_hip_y: float,
    mid_ankle_y: float,
    left_knee_ok: bool,
    right_knee_ok: bool,
    left_knee_y: float,
    right_knee_y: float,
    left_ankle_ok: bool,
    right_ankle_ok: bool,
    left_shoulder_ok: bool,
    right_shoulder_ok: bool,
    left_shoulder_x: float,
    right_shoulder_x: float,
    left_shoulder_y: float,
    right_shoulder_y: float,
    arm_left_flat: bool,
    arm_right_flat: bool,
    wrists_low_level: bool,
) -> bool:
    """Prone cues using multiple keypoints: head cluster, ankles, knees, shoulders, arms."""
    if not (lower_body_anchor_ok and horizontal_torso and hips_low):
        return False
    head_signal_ok = nose_ok or head_point_count > 0
    head_up = head_signal_ok and (shoulder_mid_y - head_ref_y) >= HEAD_CLEAR_ABOVE_SHOULDER_MIN
    chin_down = head_signal_ok and (head_ref_y - shoulder_mid_y) >= HEAD_DROPPED_BELOW_SHOULDER_MIN
    ankles = left_ankle_ok and right_ankle_ok and mid_ankle_y >= ANKLE_FLOOR_Y
    partial_leg = False
    if left_knee_ok and abs(center_hip_y - left_knee_y) < PARTIAL_LEG_HIP_KNEE_MAX_DY:
        partial_leg = True
    if right_knee_ok and abs(center_hip_y - right_knee_y) < PARTIAL_LEG_HIP_KNEE_MAX_DY:
        partial_leg = True
    arms_spread = (
        left_shoulder_ok
        and right_shoulder_ok
        and abs(left_shoulder_x - right_shoulder_x) >= SHOULDER_SPREAD_MIN_X
        and abs(left_shoulder_y - right_shoulder_y) < SHOULDER_LINE_MAX_DY
    )
    arms_along_floor = arm_left_flat or arm_right_flat
    return (
        head_up
        or chin_down
        or ankles
        or partial_leg
        or arms_spread
        or arms_along_floor
        or wrists_low_level
    )


def _resolve_fall_heuristics(
    *,
    torso_reliable: bool,
    lower_body_anchor_ok: bool,
    legs_horizontal_bilateral: bool,
    horizontal_torso: bool,
    compact_vertical_upper: bool,
    legs_horizontal: bool,
    near_floor: bool,
    hips_low: bool,
    wide_bbox: bool,
    nose_ok: bool,
    head_ref_y: float,
    head_point_count: int,
    shoulder_mid_y: float,
    center_hip_y: float,
    mid_ankle_y: float,
    left_knee_ok: bool,
    right_knee_ok: bool,
    left_knee_y: float,
    right_knee_y: float,
    left_ankle_ok: bool,
    right_ankle_ok: bool,
    left_shoulder_ok: bool,
    right_shoulder_ok: bool,
    left_shoulder_x: float,
    right_shoulder_x: float,
    left_shoulder_y: float,
    right_shoulder_y: float,
    arm_left_flat: bool,
    arm_right_flat: bool,
    wrists_low_level: bool,
) -> tuple[str, str]:
    if not torso_reliable:
        return "HUMAN_DETECTED", "Human detected with no fall posture."
    compact_for_fall = compact_vertical_upper and lower_body_anchor_ok
    wide_for_fall = wide_bbox and lower_body_anchor_ok
    floor_for_fall = near_floor and lower_body_anchor_ok
    legs_relaxed_ok = legs_horizontal and lower_body_anchor_ok and (
        legs_horizontal_bilateral or hips_low or near_floor
    )
    laydown_pose = horizontal_torso and (compact_for_fall or legs_relaxed_ok)
    fall_floor = horizontal_torso and floor_for_fall
    fall_low_scene = horizontal_torso and hips_low and (
        compact_for_fall or legs_relaxed_ok or wide_for_fall
    )
    fall_wide = wide_for_fall and horizontal_torso
    extra = _extra_prone_cues(
        lower_body_anchor_ok=lower_body_anchor_ok,
        horizontal_torso=horizontal_torso,
        hips_low=hips_low,
        nose_ok=nose_ok,
        head_ref_y=head_ref_y,
        head_point_count=head_point_count,
        shoulder_mid_y=shoulder_mid_y,
        center_hip_y=center_hip_y,
        mid_ankle_y=mid_ankle_y,
        left_knee_ok=left_knee_ok,
        right_knee_ok=right_knee_ok,
        left_knee_y=left_knee_y,
        right_knee_y=right_knee_y,
        left_ankle_ok=left_ankle_ok,
        right_ankle_ok=right_ankle_ok,
        left_shoulder_ok=left_shoulder_ok,
        right_shoulder_ok=right_shoulder_ok,
        left_shoulder_x=left_shoulder_x,
        right_shoulder_x=right_shoulder_x,
        left_shoulder_y=left_shoulder_y,
        right_shoulder_y=right_shoulder_y,
        arm_left_flat=arm_left_flat,
        arm_right_flat=arm_right_flat,
        wrists_low_level=wrists_low_level,
    )
    is_fall = laydown_pose or fall_floor or fall_low_scene or fall_wide or extra
    if is_fall:
        return "FALL_DETECTED", "Human lay-down/fall posture detected. Alert staff immediately."
    return "HUMAN_DETECTED", "Human detected with no fall posture."


def _display_model_path(candidate: str) -> str:
    try:
        path = Path(candidate)
        if path.is_file():
            return str(path.resolve())
    except OSError:
        pass
    return candidate


@dataclass(frozen=True)
class RuntimeDevice:
    device_name: str
    is_cuda: bool


class FallDetectorService:
    def __init__(self) -> None:
        self.runtime = self._resolve_runtime_device()
        self._configure_runtime()
        self.loaded_model_source: str | None = None
        self._model_is_tensorrt: bool = False
        self._model_is_onnx: bool = False
        self.model = self._load_model(settings.model_path, self.runtime.device_name)
        self._warmup_model()
        self._log_gpu_memory_after_load()

    def _resolve_runtime_device(self) -> RuntimeDevice:
        is_cuda = bool(torch.cuda.is_available())
        if settings.require_cuda and not is_cuda:
            logger.error("CUDA is required but not available. Inference is disabled.")
        if not is_cuda:
            return RuntimeDevice(device_name="cpu", is_cuda=False)
        dev_id = max(0, int(settings.cuda_device))
        count = torch.cuda.device_count()
        if count > 0 and dev_id >= count:
            logger.warning("CUDA_DEVICE=%s invalid (only %s GPU(s)); using 0.", dev_id, count)
            dev_id = 0
        return RuntimeDevice(device_name=f"cuda:{dev_id}", is_cuda=True)

    def _configure_runtime(self) -> None:
        if not self.runtime.is_cuda:
            return
        try:
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            logger.warning("CUDA runtime optimization setup failed.", exc_info=True)

    def _use_half_precision(self) -> bool:
        if self._model_is_tensorrt or self._model_is_onnx:
            return False
        return (
            self.runtime.is_cuda
            and bool(settings.use_half_precision)
            and not bool(settings.use_fp32_gpu_inference)
        )

    def _log_gpu_memory_after_load(self) -> None:
        if not self.runtime.is_cuda or self.model is None:
            return
        try:
            device = torch.device(self.runtime.device_name)
            torch.cuda.synchronize(device)
            free_b, total_b = torch.cuda.mem_get_info(device)
            reserved = torch.cuda.memory_reserved(device)
            logger.info(
                "GPU memory after load: reserved %.0f MiB, free %.0f / total %.0f MiB.",
                reserved / (1024**2),
                free_b / (1024**2),
                total_b / (1024**2),
            )
        except Exception:
            logger.debug("GPU memory snapshot skipped.", exc_info=True)

    def _try_build_tensorrt_engine_if_needed(self, model_path: str) -> None:
        if not bool(settings.tensorrt_auto_export_if_missing):
            return
        if not self.runtime.is_cuda:
            logger.warning("Skipping TensorRT auto-export: CUDA not available.")
            return
        raw = (model_path or "").strip()
        if not raw.lower().endswith(".engine"):
            return
        models_dir = Path(__file__).resolve().parents[2] / "models"
        filename = Path(raw).name
        stem = Path(filename).stem
        local_path = Path(raw)

        def user_engine_path() -> Path:
            return local_path if local_path.is_absolute() else Path.cwd() / local_path

        destinations = list(
            dict.fromkeys(
                [
                    user_engine_path(),
                    models_dir / filename,
                ]
            )
        )
        if any(p.is_file() for p in destinations):
            return

        onnx_path = models_dir / f"{stem}.onnx"
        if onnx_path.is_file() and not bool(settings.tensorrt_retry_export_after_onnx):
            print(
                f"[fall-detection] TensorRT export skipped ({onnx_path.name} present). "
                "That ONNX file can be loaded for inference (middle tier). "
                "Retry engine build: delete the .onnx and set TENSORRT_RETRY_EXPORT_AFTER_ONNX=true.",
                flush=True,
            )
            logger.info(
                "TensorRT auto-export skipped; ONNX at %s. Retry: delete file or TENSORRT_RETRY_EXPORT_AFTER_ONNX=true.",
                onnx_path,
            )
            return

        pt_file = models_dir / f"{stem}.pt"
        pt_source = str(pt_file) if pt_file.is_file() else f"{stem}.pt"
        dev = max(0, int(settings.cuda_device))
        try:
            print(
                f"[fall-detection] No .engine file; building TensorRT engine from {pt_source} "
                f"(imgsz={settings.inference_image_size}, device={dev}) — first run may take several minutes...",
                flush=True,
            )
            logger.info("TensorRT auto-export starting from %s", pt_source)
            export_model = YOLO(pt_source)
            exported = export_model.export(
                format="engine",
                imgsz=int(settings.inference_image_size),
                device=dev,
                half=not bool(settings.use_fp32_gpu_inference),
            )
            built = Path(str(exported))
            if not built.is_file():
                logger.error("TensorRT export returned invalid path: %s", exported)
                return
            for dest in destinations:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built, dest)
                logger.info("TensorRT engine installed at %s", dest.resolve())
            print(f"[fall-detection] TensorRT engine saved: {destinations[-1].resolve()}", flush=True)
        except Exception:
            logger.exception("TensorRT auto-export failed.")
            print(
                "[fall-detection] TensorRT build failed (logs often show CUDA init error 35: TensorRT vs driver/CUDA mismatch). "
                "PyTorch loads next when TENSORRT_ALLOW_PYTORCH_FALLBACK=true (default).",
                flush=True,
            )

    def _load_model(self, model_path: str, device_name: str) -> YOLO | None:
        self._try_build_tensorrt_engine_if_needed(model_path)
        candidates = self._resolve_model_candidates(model_path)
        for candidate in candidates:
            try:
                model = YOLO(candidate)
                model.to(device_name)
                self._model_is_tensorrt = str(candidate).lower().endswith(".engine")
                self._model_is_onnx = str(candidate).lower().endswith(".onnx")
                self._optimize_loaded_model(model)
                self.loaded_model_source = candidate
                if self._model_is_tensorrt:
                    kind = "TensorRT engine"
                elif self._model_is_onnx:
                    kind = "ONNX"
                else:
                    kind = "PyTorch weights"
                shown = _display_model_path(candidate)
                banner = (
                    f"{kind} | path={shown} | device={device_name} "
                    f"| imgsz={settings.inference_image_size} | requested={settings.model_path}"
                )
                print(f"[fall-detection] Using {banner}", flush=True)
                logger.info("Using inference backend: %s", banner)
                return model
            except Exception:
                logger.warning("Model candidate failed: %s", candidate, exc_info=True)
        fail_msg = (
            "No model loaded (all candidates failed). "
            f"Requested MODEL_PATH={settings.model_path!r}. Detection will be degraded."
        )
        print(f"[fall-detection] {fail_msg}", flush=True)
        logger.error(fail_msg)
        self._model_is_tensorrt = False
        self._model_is_onnx = False
        return None

    def _optimize_loaded_model(self, model: YOLO) -> None:
        if not self.runtime.is_cuda or self._model_is_tensorrt or self._model_is_onnx:
            return
        try:
            model.fuse()
        except Exception:
            logger.warning("Model fusion skipped due to runtime incompatibility.", exc_info=True)

    def _warmup_model(self) -> None:
        if self.model is None:
            return
        warmup_runs = max(0, int(settings.model_warmup_runs))
        if warmup_runs < 1:
            return
        try:
            warmup_frame = np.zeros(
                (settings.inference_image_size, settings.inference_image_size, 3),
                dtype=np.uint8,
            )
            for _ in range(warmup_runs):
                self.model.predict(
                    source=warmup_frame,
                    conf=settings.confidence_threshold,
                    verbose=False,
                    device=self.runtime.device_name,
                    imgsz=settings.inference_image_size,
                    max_det=settings.max_people_per_frame,
                    half=self._use_half_precision(),
                )
        except Exception:
            logger.warning("Model warmup skipped because it failed.", exc_info=True)

    def _resolve_model_candidates(self, model_path: str) -> list[str]:
        """TensorRT .engine must exist locally; missing engine falls back to Ultralytics hub .pt (auto-download)."""
        default_name = "yolo26n-pose.engine"
        raw = (model_path or f"models/{default_name}").strip()
        local_path = Path(raw)
        models_dir = Path(__file__).resolve().parents[2] / "models"
        filename = local_path.name if local_path.name else default_name
        stem = Path(filename).stem

        def resolve_user_path() -> Path:
            return local_path if local_path.is_absolute() else Path.cwd() / local_path

        seen: set[str] = set()
        ordered: list[str] = []

        def add(path_str: str) -> None:
            if path_str in seen:
                return
            seen.add(path_str)
            ordered.append(path_str)

        if filename.lower().endswith(".engine"):
            found_engine = False
            for p in (resolve_user_path(), models_dir / filename):
                if p.is_file():
                    found_engine = True
                    add(str(p))
            if not found_engine:
                onnx_side = resolve_user_path().with_suffix(".onnx")
                if onnx_side.is_file():
                    add(str(onnx_side))
                onnx_models = models_dir / f"{stem}.onnx"
                if onnx_models.is_file():
                    add(str(onnx_models))
            if not found_engine and bool(settings.tensorrt_allow_pytorch_fallback):
                pt_in_models = models_dir / f"{stem}.pt"
                if pt_in_models.is_file():
                    add(str(pt_in_models))
                add(f"{stem}.pt")
                logger.info(
                    "No .engine; falling back to PyTorch '%s' (TENSORRT_ALLOW_PYTORCH_FALLBACK=true).",
                    f"{stem}.pt",
                )
            elif not found_engine:
                logger.warning(
                    "No TensorRT .engine found and pytorch fallback disabled "
                    "(set TENSORRT_ALLOW_PYTORCH_FALLBACK=true or place %s).",
                    filename,
                )
            return ordered

        for p in (resolve_user_path(), models_dir / filename):
            if p.is_file():
                add(str(p))
        add(filename)
        return ordered

    def get_health(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "model_loaded": self.model is not None,
            "gpu_available": self.runtime.is_cuda,
            "device": self.runtime.device_name,
            "model_path_requested": settings.model_path,
            "model_loaded_from": self.loaded_model_source,
            "inference_image_size": settings.inference_image_size,
            "use_fp32_gpu_inference": bool(settings.use_fp32_gpu_inference),
            "tensorrt_engine": bool(self._model_is_tensorrt),
            "inference_backend": (
                "tensorrt"
                if self._model_is_tensorrt
                else "onnx"
                if self._model_is_onnx
                else "pytorch"
                if self.model is not None
                else "none"
            ),
        }
        if self.runtime.is_cuda:
            try:
                device = torch.device(self.runtime.device_name)
                torch.cuda.synchronize(device)
                free_b, total_b = torch.cuda.mem_get_info(device)
                data["gpu_mem_total_mb"] = round(total_b / (1024**2), 1)
                data["gpu_mem_free_mb"] = round(free_b / (1024**2), 1)
                data["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024**2), 1)
            except Exception:
                pass
        return data

    def analyze(self, payload: AnalyzeRequest) -> DetectionResult:
        if settings.require_cuda and not self.runtime.is_cuda:
            return DetectionResult(
                patient_id=payload.patient_id,
                area_name=payload.area_name,
                status="NO_PERSON",
                used_cuda=False,
                device="cpu",
                message="CUDA is required but not available. Install CUDA-enabled PyTorch.",
            )
        frame = self._decode_base64_frame(payload.image_base64)
        if frame is None:
            return DetectionResult(
                patient_id=payload.patient_id,
                area_name=payload.area_name,
                status="NO_PERSON",
                used_cuda=self.runtime.is_cuda,
                device=self.runtime.device_name,
                message="Invalid image format.",
            )
        if self.model is None:
            return DetectionResult(
                patient_id=payload.patient_id,
                area_name=payload.area_name,
                status="NO_PERSON",
                used_cuda=self.runtime.is_cuda,
                device=self.runtime.device_name,
                message="Detection model unavailable.",
            )

        return self._run_pose_inference(frame, payload)

    def _decode_base64_frame(self, image_base64: str) -> np.ndarray | None:
        encoded = (image_base64 or "").split(",")[-1]
        if not encoded:
            return None
        try:
            image_bytes = base64.b64decode(encoded)
            np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
            return cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        except Exception:
            logger.exception("Could not decode incoming base64 frame.")
            return None

    def _run_pose_inference(self, frame: np.ndarray, payload: AnalyzeRequest) -> DetectionResult:
        try:
            infer = self.model.predict(
                source=frame,
                conf=settings.confidence_threshold,
                verbose=False,
                device=self.runtime.device_name,
                imgsz=settings.inference_image_size,
                max_det=settings.max_people_per_frame,
                half=self._use_half_precision(),
            )
        except Exception:
            logger.exception("Pose inference failed.")
            return DetectionResult(
                patient_id=payload.patient_id,
                area_name=payload.area_name,
                status="NO_PERSON",
                used_cuda=self.runtime.is_cuda,
                device=self.runtime.device_name,
                message="Temporary detection issue.",
            )

        result = infer[0] if infer else None
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) < 1:
            return DetectionResult(
                patient_id=payload.patient_id,
                area_name=payload.area_name,
                status="NO_PERSON",
                used_cuda=self.runtime.is_cuda,
                device=self.runtime.device_name,
            )

        return self._classify_all_people(result=result, payload=payload)

    def _classify_all_people(self, result: Any, payload: AnalyzeRequest) -> DetectionResult:
        if not result or not result.keypoints or result.keypoints.xyn is None:
            return DetectionResult(
                patient_id=payload.patient_id,
                area_name=payload.area_name,
                status="NO_PERSON",
                used_cuda=self.runtime.is_cuda,
                device=self.runtime.device_name,
            )

        keypoints_list = result.keypoints.xyn.cpu().numpy()
        confidences: list[float] = []
        if result.boxes and result.boxes.conf is not None:
            confidences = result.boxes.conf.cpu().numpy().tolist()
        boxes: list[np.ndarray] = []
        if result.boxes and result.boxes.xyxyn is not None:
            boxes = result.boxes.xyxyn.cpu().numpy().tolist()

        kpt_conf_list: list[Any] = []
        conf_tensor = getattr(result.keypoints, "conf", None)
        if conf_tensor is not None:
            kpt_conf_list = conf_tensor.cpu().numpy().tolist()

        people: list[PersonDetection] = []
        for index, xy in enumerate(keypoints_list):
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            box = boxes[index] if index < len(boxes) else None
            bbox_top_left = Point(x=float(box[0]), y=float(box[1])) if box else None
            bbox_bottom_right = Point(x=float(box[2]), y=float(box[3])) if box else None
            kpt_conf = np.array(kpt_conf_list[index], dtype=np.float32) if index < len(kpt_conf_list) else None
            status, _message = self._classify_single_pose(xy=xy, box=box, kpt_conf=kpt_conf)
            people.append(
                PersonDetection(
                    person_id=index + 1,
                    status=status,
                    confidence=confidence,
                    bbox=bbox_top_left,
                    bbox_bottom_right=bbox_bottom_right,
                    keypoints=self._xy_to_keypoints(xy),
                )
            )

        highest = self._select_highest_priority_person(people)
        if highest is None:
            return DetectionResult(
                patient_id=payload.patient_id,
                area_name=payload.area_name,
                status="NO_PERSON",
                used_cuda=self.runtime.is_cuda,
                device=self.runtime.device_name,
                message="No person detected.",
            )

        message = self._status_to_message(highest.status, len(people))
        return DetectionResult(
            patient_id=payload.patient_id,
            area_name=payload.area_name,
            status=highest.status,
            confidence=highest.confidence,
            used_cuda=self.runtime.is_cuda,
            message=message,
            bbox=highest.bbox,
            bbox_bottom_right=highest.bbox_bottom_right,
            person_count=len(people),
            people=people,
            device=self.runtime.device_name,
        )

    def _xy_to_keypoints(self, xy: np.ndarray) -> list[Point]:
        points: list[Point] = []
        for i in range(int(xy.shape[0])):
            px = float(xy[i][0])
            py = float(xy[i][1])
            points.append(Point(x=max(0.0, min(1.0, px)), y=max(0.0, min(1.0, py))))
        return points

    def _classify_single_pose(
        self,
        xy: np.ndarray,
        box: np.ndarray | None = None,
        kpt_conf: np.ndarray | None = None,
    ) -> tuple[str, str]:
        kpt_min = max(0.0, min(1.0, float(settings.keypoint_confidence_min)))

        def conf_ok(idx: int) -> bool:
            if kpt_conf is None or idx >= len(kpt_conf):
                return True
            return float(kpt_conf[idx]) >= kpt_min

        def joint_usable(idx: int) -> bool:
            if not conf_ok(idx):
                return False
            px = float(xy[idx][0])
            py = float(xy[idx][1])
            return px > 1e-3 or py > 1e-3

        nose = Point(x=float(xy[NOSE_INDEX][0]), y=float(xy[NOSE_INDEX][1]))
        left_shoulder = Point(x=float(xy[LEFT_SHOULDER_INDEX][0]), y=float(xy[LEFT_SHOULDER_INDEX][1]))
        right_shoulder = Point(x=float(xy[RIGHT_SHOULDER_INDEX][0]), y=float(xy[RIGHT_SHOULDER_INDEX][1]))
        left_hip = Point(x=float(xy[LEFT_HIP_INDEX][0]), y=float(xy[LEFT_HIP_INDEX][1]))
        right_hip = Point(x=float(xy[RIGHT_HIP_INDEX][0]), y=float(xy[RIGHT_HIP_INDEX][1]))
        center_shoulder = Point(x=(left_shoulder.x + right_shoulder.x) / 2, y=(left_shoulder.y + right_shoulder.y) / 2)
        center_hip = Point(x=(left_hip.x + right_hip.x) / 2, y=(left_hip.y + right_hip.y) / 2)

        head_ref_y, head_point_count = _head_reference_y(xy, joint_usable, nose.y)
        torso_dy = abs(center_shoulder.y - center_hip.y)
        horizontal_torso = torso_dy < TORSO_HORIZONTAL_MAX_DY

        span_ys: list[float] = [head_ref_y, center_shoulder.y, center_hip.y]
        for idx in (LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, LEFT_HIP_INDEX, RIGHT_HIP_INDEX):
            if joint_usable(idx):
                span_ys.append(float(xy[idx][1]))
        vertical_span_multi = max(span_ys) - min(span_ys)
        compact_vertical_upper = vertical_span_multi < HEAD_TO_HIP_VERTICAL_SPAN_MAX

        near_floor = center_hip.y > 0.72
        hips_low = center_hip.y > HIP_LOW_Y

        left_knee = Point(x=float(xy[LEFT_KNEE_INDEX][0]), y=float(xy[LEFT_KNEE_INDEX][1]))
        right_knee = Point(x=float(xy[RIGHT_KNEE_INDEX][0]), y=float(xy[RIGHT_KNEE_INDEX][1]))
        left_ankle = Point(x=float(xy[LEFT_ANKLE_INDEX][0]), y=float(xy[LEFT_ANKLE_INDEX][1]))
        right_ankle = Point(x=float(xy[RIGHT_ANKLE_INDEX][0]), y=float(xy[RIGHT_ANKLE_INDEX][1]))
        mid_knee_y = (left_knee.y + right_knee.y) / 2
        mid_ankle_y = (left_ankle.y + right_ankle.y) / 2
        legs_horizontal_bilateral = False
        if (
            joint_usable(LEFT_KNEE_INDEX)
            and joint_usable(RIGHT_KNEE_INDEX)
            and joint_usable(LEFT_HIP_INDEX)
            and joint_usable(RIGHT_HIP_INDEX)
            and joint_usable(LEFT_ANKLE_INDEX)
            and joint_usable(RIGHT_ANKLE_INDEX)
        ):
            legs_horizontal_bilateral = (
                abs(center_hip.y - mid_knee_y) < LEG_HORIZONTAL_MAX_DY
                and abs(mid_knee_y - mid_ankle_y) < LEG_HORIZONTAL_MAX_DY
            )
        legs_horizontal = legs_horizontal_bilateral
        if not legs_horizontal:
            legs_horizontal = _single_leg_chain_horizontal(
                xy, joint_usable, LEFT_HIP_INDEX, LEFT_KNEE_INDEX, LEFT_ANKLE_INDEX
            ) or _single_leg_chain_horizontal(
                xy, joint_usable, RIGHT_HIP_INDEX, RIGHT_KNEE_INDEX, RIGHT_ANKLE_INDEX
            )

        torso_reliable = (
            joint_usable(LEFT_SHOULDER_INDEX)
            and joint_usable(RIGHT_SHOULDER_INDEX)
            and joint_usable(LEFT_HIP_INDEX)
            and joint_usable(RIGHT_HIP_INDEX)
        )
        shoulder_band_y = center_shoulder.y - ANCHOR_MAX_Y_ABOVE_SHOULDER

        def lower_anchor_plausible(idx: int) -> bool:
            if not joint_usable(idx):
                return False
            return float(xy[idx][1]) >= shoulder_band_y

        lower_body_anchor_ok = (
            lower_anchor_plausible(LEFT_KNEE_INDEX)
            or lower_anchor_plausible(RIGHT_KNEE_INDEX)
            or lower_anchor_plausible(LEFT_ANKLE_INDEX)
            or lower_anchor_plausible(RIGHT_ANKLE_INDEX)
        )

        arm_left_flat = _arm_chain_horizontal(
            xy, joint_usable, LEFT_SHOULDER_INDEX, LEFT_ELBOW_INDEX, LEFT_WRIST_INDEX
        )
        arm_right_flat = _arm_chain_horizontal(
            xy, joint_usable, RIGHT_SHOULDER_INDEX, RIGHT_ELBOW_INDEX, RIGHT_WRIST_INDEX
        )
        wrists_low_level = _wrists_low_and_level(
            xy, joint_usable, WRIST_LOW_Y, WRIST_PAIR_MAX_DY
        )

        wide_bbox = False
        if box is not None and len(box) >= 4:
            bw = float(abs(box[2] - box[0]))
            bh = float(abs(box[3] - box[1]))
            if bh > 1e-6 and bw / bh >= BBOX_WIDE_RATIO_MIN:
                wide_bbox = True

        return _resolve_fall_heuristics(
            torso_reliable=torso_reliable,
            lower_body_anchor_ok=lower_body_anchor_ok,
            legs_horizontal_bilateral=legs_horizontal_bilateral,
            horizontal_torso=horizontal_torso,
            compact_vertical_upper=compact_vertical_upper,
            legs_horizontal=legs_horizontal,
            near_floor=near_floor,
            hips_low=hips_low,
            wide_bbox=wide_bbox,
            nose_ok=joint_usable(NOSE_INDEX),
            head_ref_y=head_ref_y,
            head_point_count=head_point_count,
            shoulder_mid_y=center_shoulder.y,
            center_hip_y=center_hip.y,
            mid_ankle_y=mid_ankle_y,
            left_knee_ok=joint_usable(LEFT_KNEE_INDEX),
            right_knee_ok=joint_usable(RIGHT_KNEE_INDEX),
            left_knee_y=left_knee.y,
            right_knee_y=right_knee.y,
            left_ankle_ok=joint_usable(LEFT_ANKLE_INDEX),
            right_ankle_ok=joint_usable(RIGHT_ANKLE_INDEX),
            left_shoulder_ok=joint_usable(LEFT_SHOULDER_INDEX),
            right_shoulder_ok=joint_usable(RIGHT_SHOULDER_INDEX),
            left_shoulder_x=left_shoulder.x,
            right_shoulder_x=right_shoulder.x,
            left_shoulder_y=left_shoulder.y,
            right_shoulder_y=right_shoulder.y,
            arm_left_flat=arm_left_flat,
            arm_right_flat=arm_right_flat,
            wrists_low_level=wrists_low_level,
        )

    def _select_highest_priority_person(self, people: list[PersonDetection]) -> PersonDetection | None:
        if not people:
            return None
        return sorted(
            people,
            key=lambda item: (STATUS_PRIORITY.get(item.status, 0), item.confidence),
            reverse=True,
        )[0]

    def _status_to_message(self, status: str, person_count: int) -> str:
        if status == "FALL_DETECTED":
            return f"Human fall detected. {person_count} person(s) tracked."
        if status == "HUMAN_DETECTED":
            return f"Human detected with no fall. {person_count} person(s) tracked."
        return "No person detected."
