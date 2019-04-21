#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

cuda = torch.cuda.is_available()


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
