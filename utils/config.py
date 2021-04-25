import torch

class Config:
  	def __init__(self):

	    #Path to model checkpoint
	    self.model_path = './checkpoints/FaceGeneratorModel.pth'

	    #Folder for saving generated images
	    self.save_path = './data/res/'

	    #Image format
	    self.save_format = 'PNG'

	    #Dimension of latent space 
	    self.latent_size = 100

	    #Number of generated images 
	    self.num_generated_images = 5

	    #Device
	    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	   	#Size of generated images. Don't change this parameter.
	    self.img_size = 64

	    #Path to training dataset 
	    self.data_path = './data/dataset/'

	    #Continue training
	    self.continue_training = False

	    #Path to model to continue training
	    self.train_model_path = ''

	    #Batch size
	    self.batch_size = 64

	    #Number of learning epochs
	    self.num_epoch = 5

	    #Logging frequency
	    self.log_iter = 50

	    #Learning rate
	    self.lr = 0.0002

	    #Flip labels sometimes while training generator
	    self.flip_labels = True

	    #Generate 5 images after n epochs
	    self.make_img_samples = 1

	    #Path to generated images while training
	    self.img_samples_path = './data/train_samples/'