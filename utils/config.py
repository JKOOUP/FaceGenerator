import torch

class Config():
  	def __init__(self):

	    #Path to model checkpoint
	    self.model_path = './checkpoints/model_9.pth'

	    #Folder for saving results
	    self.save_path = './data/res/'

	    #Image format
	    self.save_format = 'PNG'

	    #Device
	    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	    #Dimension of latent space 
	    self.latent_size = 100

	    #Train sample size
	    self.img_size = 64

	    #Number of images in one batch
	    self.batch_size = 32

	    #Number of learning epochs
	    self.num_epoch = 1

	    #Log printing frequency
	    self.log_iter = 10

	    #Path to data
	    self.data_path = './data/dataset/'

	    #Learning rate
	    self.lr = 0.002