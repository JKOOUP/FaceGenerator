import os
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset
from utils.config import Config

config = Config()

class CelebADataset(Dataset):
	def __init__(self, img_path=config.data_path, transformer=None):
		
		self.img_path = img_path
		self.transformer = transformer

		self.dataset_size = 0
		for obj in os.listdir(self.img_path):	
			if os.path.isfile(os.path.join(self.img_path, obj)):
				self.dataset_size += 1

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, idx):
		img_name = str(idx + 1).zfill(6) + '.jpg'
		img = Image.open(self.img_path + img_name).convert('RGB')

		prepare_img = transforms.Compose([
			transforms.Resize(config.img_size),
    		transforms.CenterCrop(config.img_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		tensor_img = prepare_img(img)

		if self.transformer is not None:
			tensor_img = self.transformer(tensor_img)

		return tensor_img