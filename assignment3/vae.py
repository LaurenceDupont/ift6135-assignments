#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import torch.utils.data as data_utils
from torchvision.utils import save_image

def get_data_loader(dataset_location, batch_size):
    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        dataset = data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata


MINI_BATCH_SIZE = 128

train, valid, test = get_data_loader("datasets/binarized_mnist", MINI_BATCH_SIZE)

#%%

# Based on the "Vanilla VAE code snippet" seen in class: https://chinweihuang.files.wordpress.com/2019/04/vae_lecture_2019_full.pdf

class VAE(nn.Module):
    
    DIMENSION_H = 256
    DIMENSION_Z = 100
    
    def __init__(self):
        super(VAE, self).__init__()
        
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ELU(),
            
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(64, 256, 5),
            nn.ELU()
        )
        
        self.encoder_fc = nn.Linear(self.DIMENSION_H, self.DIMENSION_Z * 2)
        
        self.decoder_fc = nn.Linear(self.DIMENSION_Z, self.DIMENSION_H)

        self.decoder = nn.Sequential(
            nn.ELU(),
            
            nn.Conv2d(256, 64, 5, padding=4),
            nn.ELU(),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=2),
            nn.ELU(),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=2),   
            nn.ELU(),
            
            nn.Conv2d(16, 1, 3, padding=2)
        )
        
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
        x_ = self.decoder(self.decoder_fc(z).unsqueeze(2).unsqueeze(3))
        
        return x_, mu, log_sigma

    # References:
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    # https://github.com/Lasagne/Recipes/blob/master/examples/variational_autoencoder/variational_autoencoder.py
    def loss_function(self, x_, x, mu, log_sigma):
        mini_batch_size = x_.size(0)
        
        BCE = -self.bce(x_.squeeze().view(-1, 784), x.squeeze().view(-1, 784))
        
        KLD = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp())
    
        return -(BCE - KLD) / mini_batch_size


def evaluate_elbo_loss(vae, dataset): # Per-instance ELBO
    with torch.no_grad():
        vae.eval()
        
        mini_batches_count = len(dataset)
        
        total_loss = 0
        
        for batch_index, batch in enumerate(dataset):
            x = batch[0]
            if cuda:
                x = x.cuda()

            x_, mu, log_sigma = vae(x)
            
            loss = vae.loss_function(x_, x, mu, log_sigma)
            total_loss += loss.item()
            
        return total_loss / mini_batches_count


def estimate_log_likelihood_minibatch(vae, minibatch, K=200): # K is the number of samples
    with torch.no_grad():
        vae.eval()
        
        log_likelihoods = []
    
        for item in minibatch:
            x = item
            if cuda:
                x = x.cuda()
    
            samples_elbos = []
            
            for sample_index in range(K):
                x_, mu, log_sigma = vae(x)
                                    
                sample_elbo = -vae.loss_function(x_, x, mu, log_sigma)
                samples_elbos.append(sample_elbo)
                
            samples_elbos = torch.stack(samples_elbos)
            
            pi = torch.max(samples_elbos)
    
            log_likelihood = pi + torch.log(torch.sum(torch.exp(samples_elbos - pi))) - np.log(K)
            log_likelihoods.append(log_likelihood.item())
        
        return log_likelihoods
    

def estimate_log_likelihood(vae, dataset):
    with torch.no_grad():
        vae.eval()
        
        total_log_likelihoods = 0
        
        for batch_index, batch in enumerate(dataset):
                minibatch_log_likelihoods = estimate_log_likelihood_minibatch(vae, batch)
                total_log_likelihoods += np.sum(minibatch_log_likelihoods)
        
        return total_log_likelihoods  / len(dataset)
    

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
        
        x_, mu, log_sigma = vae(x)
        
        if batch_index == 0:
            # Source for reconstructing the image: https://github.com/pytorch/examples/blob/master/vae/main.py
            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n], F.sigmoid(x_[:n])])
            save_image(comparison.cpu(), 'vae_results/reconstruction_' + str(epoch) + '.png', nrow=n)

        loss = vae.loss_function(x_, x, mu, log_sigma)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch_index + 1) % 100 == 0:
            print(-loss.item())
            
    valid_elbo_loss = evaluate_elbo_loss(vae, valid)
    print("Validation ELBO loss:", -valid_elbo_loss)
    
    if epoch == 19:
        valid_log_likelihood = estimate_log_likelihood(vae, valid)
        print("Validation log likelihood:", valid_log_likelihood)
    
    test_elbo_loss = evaluate_elbo_loss(vae, test)
    print("Test ELBO loss:", -test_elbo_loss)
    
    if epoch == 19:
        test_log_likelihood = estimate_log_likelihood(vae, test)
        print("Test log likelihood:", test_log_likelihood)