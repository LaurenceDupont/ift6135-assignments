from __future__ import print_function
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch import autograd
from assignment3.samplers import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def jsd(p, q):
    '''
    Implementation of pairwise Jensen Shannon Divergence
    :param p: (sequence) Distribution p
    :param q: (sequence) Distribution q
    :return: (float) The calculated entropy
    '''
    p, q = np.asarray(p), np.asarray(q)
    p, q = p / p.sum(), q / q.sum()  # Normalize the distribution p
    r = (p + q) / 2.
    return np.sum(p * np.log(p / r), axis=0) / 2. + np.sum(q * np.log(q / r), axis=0) / 2.


# inspired from https://github.com/Yangyangii/GAN-Tutorial
class MLP_Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, lr=1e-3):
        super(MLP_Discriminator, self).__init__()

        self.input_size = input_size  # 512
        self.hidden_size = hidden_size  # 256

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(self.hidden_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.discriminator(x)
        return y


### Main ###
input_size = 512
hidden_size = 256
D = MLP_Discriminator(input_size, hidden_size).to(DEVICE)
batch_size = 100
sample_size = 512
max_epoch = 20
criterion = nn.BCELoss()  # METTRE LA BONNE LOSS
D_opt = torch.optim.Adam(D.parameters())
D_real = torch.ones(batch_size, 1).to(DEVICE)  # Discriminator Label to real
D_fakes = torch.zeros(batch_size, 1).to(DEVICE)  # Discriminator Label to fake

for epoch in range(max_epoch):
    x = torch.FloatTensor(batch_size, sample_size).normal_(0, 1).to(DEVICE)
    x_output = D(x)
    D_x_loss = criterion(x_output, D_real)

    # dist = iter(distribution3(sample_size))
    # z = torch.FloatTensor(next(dist)).to(DEVICE)
    z = torch.FloatTensor(batch_size, sample_size).normal_(0, 1).to(DEVICE)
    z_outputs = D(z)
    D_z_loss = criterion(z_outputs, D_fakes)
    D_loss = D_x_loss + D_z_loss  # METTRE LA BONNE LOSS

    D.zero_grad()
    D_loss.backward()
    D_opt.step()
    print('Epoch: {}/{}, D Loss: {}'.format(epoch, max_epoch, D_loss.item()))
