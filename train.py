import torch
import torch.nn as nn
from torchvision import transforms, datasets

from models.generator import Generator
from models.dataset import CelebADataset
from models.discriminator import Discriminator

from utils.utils import weights_init
from utils.config import Config

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

config = Config()

def get_dataloader(dataset):
	loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
	return loader

def D_train(data, D, G, optim_D, criterion, current_size, labels0, labels1, noise):
    D.train()
    D.zero_grad()

    data = data.to(config.device)
    outp = D(data).view(current_size)

    D_loss1 = criterion(outp, labels1)

    results = G(noise).detach()
    outp = D(results).view(current_size)

    D_loss2 = criterion(outp, labels0)

    D_loss = D_loss1 + D_loss2

    D_loss.backward()
    optim_D.step()

    return D_loss.item()

def G_train(D, G, optim_G, criterion, current_size, labels0, labels1, noise):
    G.train()
    G.zero_grad()

    results = G(noise)
    outp = D(results).view(current_size)

    if config.flip_labels == True:
    	if torch.randn(1) > 0.8:
    		G_loss = criterion(outp, labels0)
    	else: 
    		G_loss = criterion(outp, labels1)

    G_loss.backward()
    optim_G.step()

    return G_loss.item()

def log_history(epoch, iters, num_iters, D_losses, G_losses):
	print('Epoch: {}, Batch: {}/{}, G loss: {:.4f}, D loss: {:.4f}'.format(
    	epoch, iters, num_iters, G_losses[-1], D_losses[-1]
  	))

def train(loader, D, G, optim_D, optim_G, criterion):
	iters = 0
	G_losses = [0]
	D_losses = [0]

	for i in range(config.num_epoch):
		for data in loader:
			current_size = data.size(0)

			labels0 = torch.tensor([0] * current_size).to(config.device, torch.float)
			labels1 = torch.tensor([1] * current_size).to(config.device, torch.float)

			noise = torch.randn((current_size, config.latent_size, 1, 1)).to(config.device)

			D_loss = D_train(data, D, G, optim_D, criterion, current_size, labels0, labels1, noise)
			
			G_loss = G_train(D, G, optim_G, criterion, current_size, labels0, labels1, noise)

			iters += 1
			D_losses.append(D_loss)
			G_losses.append(G_loss)

			if iters % config.log_iter == 0:
				log_history(i, iters, len(loader), D_losses, G_losses)

		save_path = './checkpoints/model_{}_{:.4f}_{:.4f}.pth'.format(i, D_losses[-1], G_losses[-1])
		torch.save({
			'Generator_state_dict' : G.state_dict(),    
			'G_optim_state_dict' : optim_G.state_dict(),
			'Discriminator_state_dict' : D.state_dict(),
			'D_optim_state_dict' : optim_D.state_dict()
		}, save_path)


if __name__ == '__main__':
	dataset = CelebADataset()

	dataloader = get_dataloader(dataset)

	G = Generator(config.latent_size).to(config.device)
	G.apply(weights_init)

	D = Discriminator().to(config.device)
	D.apply(weights_init)

	optim_G = torch.optim.AdamW(G.parameters(), lr=config.lr, betas=(0.5, 0.999))
	optim_D = torch.optim.AdamW(D.parameters(), lr=config.lr, betas=(0.5, 0.999))
	criterion = nn.BCELoss()

	train(dataloader, D, G, optim_D, optim_G, criterion)
