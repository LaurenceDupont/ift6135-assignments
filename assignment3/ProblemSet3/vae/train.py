#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
from torch.optim import Adam
from torchvision.utils import save_image
import numpy as np

from vae_svhn import VAE

MINI_BATCH_SIZE = 128

def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=transforms.ToTensor()
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader


def evaluate_elbo_loss(vae, dataset): # Per-instance ELBO
    with torch.no_grad():
        vae.eval()
        
        mini_batches_count = len(dataset)
        
        total_loss = 0
        
        for batch_index, batch in enumerate(dataset):
            x = batch[0]
            if cuda:
                x = x.cuda()

            decoder_mu, mu, log_sigma = vae(x)
            
            loss = vae.loss_function(decoder_mu, x, mu, log_sigma)
            total_loss += loss.item()
            
        return total_loss / mini_batches_count
    

train, valid, test = get_data_loader("../../datasets/svhn", MINI_BATCH_SIZE)

vae = VAE()
params = vae.parameters()
optimizer = Adam(params, lr=2e-4)
cuda = torch.cuda.is_available()
if cuda:
    vae = vae.cuda()
    
best_elbo_loss = np.Inf

for epoch in range(20):
    print('epoch ' + str(epoch))
    
    vae.train()
    
    for batch_index, batch in enumerate(train):
        x = batch[0]
        if cuda:
            x = x.cuda()

        decoder_mu, mu, log_sigma = vae(x)
        
        if batch_index == 0:
            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n], decoder_mu[:n]])
            save_image(comparison.cpu(), 'reconstruction/reconstruction_' + str(epoch) + '.png', nrow=n)

        loss = vae.loss_function(decoder_mu, x, mu, log_sigma)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch_index + 1) % 100 == 0:
            print(-loss.item())
            
    valid_elbo_loss = evaluate_elbo_loss(vae, valid)
    print("Validation ELBO loss:", -valid_elbo_loss)
    
    if valid_elbo_loss < best_elbo_loss:
        best_elbo_loss = valid_elbo_loss
        torch.save(vae.state_dict(), "saved_models/vae_svhn.pt")
        print("Saved.")