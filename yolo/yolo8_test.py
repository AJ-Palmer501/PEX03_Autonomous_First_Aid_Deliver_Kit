import torch
import numpy as np
import time
from ultralytics import YOLO

print('Version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
a = torch.zeros(2).cuda()
print('GPU tensor:', a)

#from huggingface_hub import hf_hub_download
# YOLOv8s trained on VisDrone — pedestrian, people, bicycle, car, van, truck etc.
# Download the baseline yolov8s (best accuracy, ~30 GMACs)
#path = hf_hub_download(
#    repo_id='ENOT-AutoDL/yolov8s_visdrone',
#    filename='enot_neural_architecture_selection_x2/weights/best.pt')

path ="/home/usafa/usafa_472/sentinel_drone/yolo/yolov8m-visdrone.pt"

print('Downloaded to:', path)

from ultralytics import YOLO
import shutil

# Copy from HuggingFace cache to working directory
#shutil.copy(path, 'yolov8s_visdrone.pt')

#model = YOLO('yolov8s_visdrone.pt')
model = YOLO(path)
model.export(format='engine', device=0, half=True, imgsz=640)

#model = YOLO('yolov8s.pt')
#model.export(format='engine', device=0, half=True, imgsz=640)
# Expect ~25-30 FPS but meaningfully better detection of small/distant people

'''
frame = np.zeros((480, 640, 3), dtype=np.uint8)

print('--- TensorRT .engine ---')
model_trt = YOLO('yolov8n.engine')
# Warmup — first few frames are always slow
for _ in range(3):
    model_trt(frame, verbose=False)

start = time.time()
for _ in range(30):
    model_trt(frame, verbose=False)
elapsed = time.time() - start
print(f'30 frames: {elapsed:.2f}s  ->  {30/elapsed:.1f} FPS')
'''