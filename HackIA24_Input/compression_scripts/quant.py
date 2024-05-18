import torch
import torch
import numpy as np
import torch
from tqdm.auto import tqdm
assert torch.cuda.is_available()
from torch2trt import torch2trt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('models/classifier.pt').to(device)

example_input = torch.randn(1, 3, 224, 224).cuda()


model_trt = torch2trt(model,[example_input], fp16_mode=True)

torch.save(model_trt.state_dict(), 'models/quantized_classifier.pt')