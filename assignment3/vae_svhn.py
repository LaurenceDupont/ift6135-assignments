#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image

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


MINI_BATCH_SIZE = 128

train, valid, test = get_data_loader("datasets/svhn", MINI_BATCH_SIZE)

#%%

# Based on the "Vanilla VAE code snippet" seen in class: https://chinweihuang.files.wordpress.com/2019/04/vae_lecture_2019_full.pdf

class VAE(nn.Module):
    
    DIMENSION_H = 256
    DIMENSION_Z = 100
    DIM = 128
    
    def __init__(self):
        super(VAE, self).__init__()
        
        self.MSE = nn.MSELoss(reduction='sum')
        
        # Encoder
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9),
            nn.ELU(),
            
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(64, 256, 5),
            nn.ELU()
        )
        
        self.encoder_fc = nn.Linear(self.DIMENSION_H, self.DIMENSION_Z * 2)
        
        # Decoder
        
        self.preprocess = nn.Sequential(
            nn.Linear(self.DIMENSION_Z, 4 * 4 * 4 * self.DIM),
            nn.ReLU(True),
        )
        
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.DIM, 2 * self.DIM, 2, stride=2),
            nn.BatchNorm2d(2 * self.DIM),
            nn.ReLU(True),
        )
       
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.DIM, self.DIM, 2, stride=2),
            nn.BatchNorm2d(self.DIM),
            nn.ReLU(True),
        )
        
        self.deconv_out = nn.ConvTranspose2d(self.DIM, 3, 2, stride=2)
        self.sig = nn.Sigmoid()
   
    
    def decode(self, z):
        output = self.preprocess(z)
        output = output.view(-1, 4 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sig(output)
        return output.view(-1, 3, 32, 32)
    
        
    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(log_sigma) + 1e-7
        e = torch.randn(mu.size()[0], self.DIMENSION_Z)
        if cuda:
            e = e.cuda()
        z = mu + sigma * e
        
        return z
        

    def forward(self, x):
        q_parameters = self.encoder_fc(self.encoder(x).squeeze())
        mu, log_sigma = q_parameters[:, :self.DIMENSION_Z], q_parameters[:, self.DIMENSION_Z:]
        z = self.reparameterize(mu, log_sigma)
        decoder_mu = self.decode(z)
        
        return decoder_mu, mu, log_sigma

    # References:
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    # https://github.com/Lasagne/Recipes/blob/master/examples/variational_autoencoder/variational_autoencoder.py
    # https://github.com/1Konny/WAE-pytorch/blob/master/ops.py
    def loss_function(self, decoder_mu, x, mu, log_sigma):
        x_reshaped = x.reshape(-1, 3*32*32)
        decoder_mu_reshaped = decoder_mu.reshape(-1, 3*32*32)
        mini_batch_size = x_reshaped.size(0)
        
        # Compute the MSE loss
        log_px_z = -self.MSE(decoder_mu_reshaped, x_reshaped)
        
        # Compute the KL divergence
        KLD = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp())
    
        return -(log_px_z - KLD) / mini_batch_size


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

vae = VAE()
params = vae.parameters()
optimizer = Adam(params, lr=3e-4)
cuda = torch.cuda.is_available()
if cuda:
    vae = vae.cuda()

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
            save_image(comparison.cpu(), 'vae_svhn_results/reconstruction_' + str(epoch) + '.png', nrow=n)

        loss = vae.loss_function(decoder_mu, x, mu, log_sigma)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch_index + 1) % 100 == 0:
            print(-loss.item())
            
    valid_elbo_loss = evaluate_elbo_loss(vae, valid)
    print("Validation negative ELBO loss:", -valid_elbo_loss)