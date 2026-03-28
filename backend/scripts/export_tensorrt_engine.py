# Purpose: Export Ultralytics pose weights to TensorRT .engine (matches API MODEL_PATH=models/yolo26n-pose.engine).
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO pose .pt to TensorRT .engine")
    parser.add_argument(
        "--weights",
        default="yolo26n-pose.pt",
        help="Official hub name or path to .pt (e.g. yolo26n-pose.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Must match INFERENCE_IMAGE_SIZE in production .env")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Build FP32 engine (larger); default is FP16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional: copy .engine here (e.g. models/yolo26n-pose.engine)",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: ultralytics is not installed in this Python environment", file=sys.stderr)
        return 1

    try:
        model = YOLO(args.weights)
        exported = model.export(
            format="engine",
            imgsz=args.imgsz,
            device=args.device,
            half=not args.fp32,
        )
    except Exception as exc:
        print(f"error: export failed: {exc}", file=sys.stderr)
        print(
            "\nIf you see CUDA error 35 or 'factory function returned nullptr':\n"
            "  The TensorRT Python wheel must match your NVIDIA driver and CUDA version.\n"
            "  PyTorch may work (cu126) while TensorRT’s bundled CUDA does not.\n"
            "  Fix: install TensorRT for your CUDA from NVIDIA, or update the GPU driver,\n"
            "  then retry. Until then use yolo26n-pose.onnx or yolo26n-pose.pt for inference.\n",
            file=sys.stderr,
        )
        return 1

    exp_path = Path(str(exported)).resolve()
    if not exp_path.is_file():
        print(f"error: expected file at {exp_path}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exp_path, args.output)
        print(f"Copied engine to {args.output.resolve()}")
    else:
        print(f"Engine written to {exp_path}")

    print("Set MODEL_PATH to that .engine path and match INFERENCE_IMAGE_SIZE to this export imgsz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
