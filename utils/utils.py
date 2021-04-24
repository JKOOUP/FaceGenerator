import time
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

class Timer:
	def __init__(self):
		self.start_time = time.perf_counter()

		self.batch_times = [self.start_time]
		self.epoch_times = [self.start_time]

	def save_batch_time(self):
		self.batch_times.append(time.perf_counter())

	def get_last_batch_time(self):
		elapsed_time = self.batch_times[-1] - self.batch_times[-2]
		return '{:.2f}'.format(elapsed_time)

	def save_epoch_time(self):
		self.epoch_times.append(time.perf_counter())

	def get_last_epoch_time(self):
		return self.log_time(int(self.epoch_times[-1] - self.epoch_times[-2]))

	def log_time(self, curr_time):	
		return '{:d}:{:02d}:{:02d}'.format(curr_time // 3600, (curr_time % 3600) // 60, curr_time % 60)