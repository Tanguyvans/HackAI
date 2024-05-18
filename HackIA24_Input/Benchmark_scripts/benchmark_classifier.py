import torch 
from benchmark import benchmark, measure_latency_gpu
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms


from torch2trt import TRTModule
from torch2trt import torch2trt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load('../models/FireResNet50-97.pt')

input_data = torch.rand(1, 3, 224, 224).to(device)

print('----- Base Model ----- ')
benchmark(model, dummy_input=input_data)

print('----- Pruned Model ----- ')
pruned_model = torch.load('../compressed_models/FireResNet50_pruned.pt')
benchmark(pruned_model, dummy_input=input_data)

""" # this code takes a model and saves a tensorrt engine for inference  
model_trt = torch2trt(model.eval().to(device), [input_data], fp16_mode=True, max_batch_size=128)
torch.save(model_trt.state_dict(), 'compressed_models/classifierFP16.pth')
"""

print('----- Quantized model ----- ')
# benchmark(model_trt, dummy_input=input_data)
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('../compressed_models/classifierFP16.pth'))
measure_latency_gpu(model_trt, input_data)
