import torch
import numpy as np
from PIL import Image

def tensor_to_np(tensor):
	arr = tensor.squeeze(0).detach().numpy()
	arr = arr.transpose((1, 2, 0))
	return arr

def prepare_result(tensor):
	np_img = tensor_to_np(tensor)
	img = Image.fromarray(np.uint8(255 * np_img))
	return img
