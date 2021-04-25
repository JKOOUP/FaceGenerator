import os
import torch
import numpy as np
from PIL import Image

from utils.config import Config
from utils.utils import prepare_result
from models.generator import Generator 

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

config = Config()

def load_generator():
	generator = Generator(config.latent_size).to(config.device)
	model = torch.load(config.model_path, map_location=config.device)
	generator.load_state_dict(model['Generator_state_dict'])
	return generator

def get_tensor_image(latent_sample=0):
	if latent_sample == 0:
		latent_sample = generator.get_sample(1, config.device)
	result = generator(latent_sample)
	result = (result - result.min()) / (result.max() - result.min())
	return result

def save_result(result, path=config.save_path):
	count = len(os.listdir(path)) + 1

	img = prepare_result(result)
	img.save(path + str(count) + '.' + config.save_format.lower(), config.save_format)

if __name__ == '__main__':
	generator = load_generator()

	for i in range(config.num_generated_images):
		tensor_img = get_tensor_image()
		save_result(tensor_img)