from pathlib import Path
import platform

import numpy as np
import torch
from ultralytics import YOLO


repo_root = Path(__file__).resolve().parent.parent
arch = platform.machine().lower()


def select_model_path() -> Path:
    candidates = (
        [repo_root / "yolo" / "yolov8m-visdrone.engine",
         repo_root / "yolo" / "yolov8m-visdrone.pt"]
        if arch in {"aarch64", "arm64"}
        else [repo_root / "yolo" / "yolov8m-visdrone.pt",
              repo_root / "yolo" / "yolov8m-visdrone.engine"]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


model_path = select_model_path()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Torch version:", torch.__version__)
print("Architecture:", arch)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Model path:", model_path)

model = YOLO(str(model_path))
test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
results = model.predict(source=test_frame,
                        device=device,
                        imgsz=640,
                        verbose=False)

print(f"Ran {len(results)} inference batch(es) on {device}")
