import torch
import numpy as np
import torch.nn as nn

from PIL import Image

def tensor_to_np(tensor):
	arr = tensor.squeeze(0).detach().cpu().numpy()
	arr = arr.transpose((1, 2, 0))
	return arr

def prepare_result(tensor):
	np_img = tensor_to_np(tensor)
	img = Image.fromarray(np.uint8(255 * np_img))
	return img

def weights_init(model):
    classname = model.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)