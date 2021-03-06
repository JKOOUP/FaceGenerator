import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets

from models.generator import Generator
from models.dataset import CelebADataset
from models.discriminator import Discriminator

from utils.utils import Timer, weights_init, prepare_result
from utils.utils import log_batch_history, log_epoch_history
from utils.utils import save_model, load_models_with_optims
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
    outp_real = D(data).view(current_size, 2)

    D_loss1 = criterion(outp_real, labels1)

    results = G(noise).detach()
    outp_fake = D(results).view(current_size, 2)

    D_loss2 = criterion(outp_fake, labels0)

    D_loss = D_loss1 + D_loss2

    D_loss.backward()
    optim_D.step()

    return D_loss.item()

def G_train(D, G, optim_G, criterion, current_size, labels0, labels1, noise):
    G.train()
    G.zero_grad()

    results = G(noise)
    outp = D(results).view(current_size, 2)

    if config.flip_labels == True:
    	if torch.rand(1) > 0.8:
    		G_loss = criterion(outp, labels0)
    	else:
    		G_loss = criterion(outp, labels1)

    G_loss.backward()
    optim_G.step()

    return G_loss.item()

def make_img_samples(generator):
	latent_sample = generator.get_sample(1, config.device)
	result = generator(latent_sample).clamp_(0, 1)

	count = len(os.listdir(config.img_samples_path)) + 1

	img = prepare_result(result)
	img.save(config.img_samples_path + str(count) + '.' + config.save_format.lower(), config.save_format)

def train(loader, D, G, optim_D, optim_G, criterion):
	G_losses = [0]
	D_losses = [0]

	timer = Timer()

	for i in range(1, config.num_epoch + 1):
		iters = 0

		for data in loader:
			current_size = data.size(0)

			labels0 = torch.tensor([0] * current_size).to(config.device, torch.long)
			labels1 = torch.tensor([1] * current_size).to(config.device, torch.long)

			noise = torch.randn((current_size, config.latent_size, 1, 1)).to(config.device)

			D_loss = D_train(data, D, G, optim_D, criterion, current_size, labels0, labels1, noise)
			G_loss = G_train(D, G, optim_G, criterion, current_size, labels0, labels1, noise)

			iters += 1
			D_losses.append(D_loss)
			G_losses.append(G_loss)			

			if iters % config.log_iter == 0:
				timer.save_batch_time()
				log_batch_history(i, iters, len(loader), D_losses, G_losses, timer)

		save_model(i, G, optim_G, D, optim_D)

		timer.save_epoch_time()
		log_epoch_history(i, len(loader), D_losses, G_losses, timer)

		if i % config.make_img_samples == 0:
			for x in range(5):
				make_img_samples(G)


if __name__ == '__main__':
	dataset = CelebADataset()

	dataloader = get_dataloader(dataset)

	G = Generator(config.latent_size).to(config.device)
	D = Discriminator().to(config.device)
	
	optim_G = torch.optim.AdamW(G.parameters(), lr=config.lr, betas=(0.5, 0.999))
	optim_D = torch.optim.AdamW(D.parameters(), lr=config.lr, betas=(0.5, 0.999))

	if (config.continue_training):
		G, optim_G, D, optim_D = load_models_with_optims(
			G, optim_G, D, optim_D, config.train_model_path, config.device
		)
	else:
		G.apply(weights_init)
		D.apply(weights_init)
	
	criterion = nn.CrossEntropyLoss()

	train(dataloader, D, G, optim_D, optim_G, criterion)
