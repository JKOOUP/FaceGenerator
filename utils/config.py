import torch

class Config():
	def __init__(self):
		
		self.model_path = './checkpoints/model_9.pth'
		self.save_path = './data/res/'
		self.save_format = 'PNG'
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.latent_size = 100