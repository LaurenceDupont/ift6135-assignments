#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np
import os
import torch.utils.data as data_utils
from torchvision.utils import save_image
#%%
# Source: https://github.com/jmtomczak/vae_vpflows/blob/master/utils/load_data.py
def get_data_loader(batch_size):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_train.amat')) as f:#open("C:\\Users\\Game\\AI\\ift6135-assignments\\assignment3\\datasets\\MNIST_static\\binarized_mnist_train.amat") as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:#open("C:\\Users\\Game\\AI\\ift6135-assignments\\assignment3\\datasets\\MNIST_static\\binarized_mnist_valid.amat") as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
#    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
#        lines = f.readlines()
#    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    #y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=batch_size, shuffle=False)

    #test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    #test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader#, test_loader

MINI_BATCH_SIZE = 128

train, valid = get_data_loader(MINI_BATCH_SIZE)

#%%
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

    # Source: https://github.com/pytorch/examples/blob/master/vae/main.py
    def loss_function(self, recon_x, x, mu, log_sigma):
        BCE = self.bce(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    
        return BCE + KLD


def evaluate(vae, dataset):
    with torch.no_grad():
        vae.eval()
        correct = 0.
        total = 0.
        for x, y in dataset:
            if cuda:
                x = x.cuda()
                y = y.cuda()

            x_, mu, log_sigma = vae(x)
            out = F.sigmoid(x)
            c = (out.argmax(dim=-1) == y).sum().item()
            t = x.size(0)
            correct += c
            total += t
    acc = correct / float(total)
    return acc

vae = VAE()
params = vae.parameters()
optimizer = Adam(params, lr=3e-4)
ce = nn.CrossEntropyLoss()
best_acc = 0.
cuda = torch.cuda.is_available()
if cuda:
    vae = vae.cuda()

for epoch in range(20):
    print('epoch ' + str(epoch))
    vae.train()
    for i, (x, y) in enumerate(train):
        if cuda:
            x = x.cuda()
            y = y.cuda()
        x = x.reshape((-1, 1, 28, 28))
        x_, mu, log_sigma = vae(x)
        
        if i == 0:
            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n], F.sigmoid(x_[:n])]) # torch.cat([x[:n], F.sigmoid(x_[:n])])
            save_image(comparison.cpu(), 'vae_results/reconstruction_' + str(epoch) + '.png', nrow=n)

        loss = vae.loss_function(x_.squeeze().view(-1, 784), x.squeeze().view(-1, 784), mu, log_sigma)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 100 == 0:
            print(loss.item())
            
    acc = evaluate(vae, valid)
    #print("Validation acc:", acc,)

    #if acc > best_acc:
    #    best_acc = acc