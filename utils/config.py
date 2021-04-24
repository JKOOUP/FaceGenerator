import torch

class Config:
  	def __init__(self):

	    #Path to model checkpoint
	    self.model_path = './checkpoints/model_9.pth'

	    #Folder for saving results
	    self.save_path = './data/res/'

	    #Path to data
	    self.data_path = './data/dataset/'

	    #Continue training
	    self.continue_training = False
	    self.train_model_path = './checkpoints/model_0_1.7589_10.8377.pth'

	    #Image format
	    self.save_format = 'PNG'

	    #Device
	    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	    #Dimension of latent space 
	    self.latent_size = 100

	    #Train sample size
	    self.img_size = 64

	    #Number of images in one batch
	    self.batch_size = 64

	    #Number of learning epochs
	    self.num_epoch = 5

	    #Log printing frequency
	    self.log_iter = 10

	    #Learning rate
	    self.lr = 0.0002

	    #Flip labels sometimes while training generator
	    self.flip_labels = True

	    #Generate 5 images after n epochs
	    self.make_img_samples = 1

	    self.img_samples_path = './data/train_samples/'
