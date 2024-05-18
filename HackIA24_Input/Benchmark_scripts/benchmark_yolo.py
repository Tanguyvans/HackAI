import torch 
from ultralytics import YOLO
import numpy as np 
import time 
from tqdm import tqdm 
from benchmark import benchmark_yolo

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# base model full precision non pruned
model_weight = '../models/best.pt'

base_model = YOLO(model_weight)

# export to tensorrt
# base_model.export(format="engine", device=device, half=True, imgsz=224)

# quantized model
engine_file = "../compressed_models/yolo_quantized.engine"
quantized_model = YOLO(engine_file)


# input data 
input_data = np.random.rand(224, 224, 3).astype(np.float32) 

print('----- Base model -----')
benchmark_yolo(base_model, input_data)

print('----- Quantized Model -----')
benchmark_yolo(quantized_model, input_data)
