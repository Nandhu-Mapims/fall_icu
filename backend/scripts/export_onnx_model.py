# Purpose: Export Ultralytics YOLO26 pose .pt to ONNX (format="onnx") for inference without TensorRT.
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO pose .pt to ONNX")
    parser.add_argument("--weights", default="yolo26n-pose.pt", help="Hub name or path to .pt")
    parser.add_argument("--imgsz", type=int, default=960, help="Match INFERENCE_IMAGE_SIZE in .env")
    parser.add_argument("--device", default="0", help="CUDA device for export")
    parser.add_argument("--output", type=Path, default=None, help="Copy .onnx here (e.g. models/yolo26n-pose.onnx)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: ultralytics is not installed", file=sys.stderr)
        return 1

    try:
        model = YOLO(args.weights)
        exported = model.export(
            format="onnx",
            imgsz=int(args.imgsz),
            device=args.device,
            simplify=True,
        )
    except Exception as exc:
        print(f"error: ONNX export failed: {exc}", file=sys.stderr)
        return 1

    exp_path = Path(str(exported)).resolve()
    if not exp_path.is_file():
        print(f"error: missing output {exp_path}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exp_path, args.output)
        print(f"ONNX copied to {args.output.resolve()}")
    else:
        print(f"ONNX written to {exp_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["main"]
