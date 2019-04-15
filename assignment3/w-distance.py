#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np
from torch import autograd
from samplers import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%

# inspired from https://github.com/jalola/improved-wgan-pytorch
class W_Discriminator(nn.Module):
    def __init__(self, lr=1e-3):
        super(W_Discriminator, self).__init__()

        self.input_size = 512
        self.hidden_size = 512
        self.Lambda = 11 # Gradient penalty lambda hyperparameter

        self.discriminator = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, 1),
            #nn.Sigmoid() # TODO I need to check if this ok...
        )

        self.optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
    

    def forward(self, x):
        y = self.discriminator(x)
        return y

    def gradient_penalty(self, x_real, x_fake):
        alpha = torch.FloatTensor(x_real.size()).uniform_(0, 1).to(DEVICE)

        z = alpha * x_real.detach() + (1 - alpha) * x_fake.detach() # Detach from other graph
        z.requires_grad_(True)

        T_z = self.forward(z) # T(z)
        
        grads = autograd.grad(outputs=T_z, inputs=z, # outputs w.r.t. the inputs
            grad_outputs=torch.ones(T_z.size()).to(DEVICE), # Keep all grads
            create_graph=True
        )[0]

        grad_penalty = ((grads.norm(2) -1) ** 2).mean() * self.Lambda
        return grad_penalty



    def loss_function(self, x_real, x_fake, forward_real, forward_fake):
        
        forward_real = forward_real.mean()
        forward_fake = forward_fake.mean()

        gradient_penalty = self.gradient_penalty(x_real, x_fake)

        d_loss = forward_real - forward_fake - gradient_penalty
        WD = forward_real - forward_fake
        return d_loss, WD

#%%

###########################
# Start of execution main
##########################

D = W_Discriminator().to(DEVICE)
mone = torch.FloatTensor([-1]).to(DEVICE)
batch_size = 100
sample_size = 512
n_critic = 10
for crit in range(n_critic):
    first_x = torch.FloatTensor(batch_size, sample_size).normal_(0,1).to(DEVICE)
    second_x = torch.FloatTensor(batch_size, sample_size).normal_(3,1).to(DEVICE)

    D.optim.zero_grad()
    x_real = D(first_x)
    x_fake = D(second_x)

    loss, WD = D.loss_function(first_x, second_x, x_real, x_fake)

    loss.backward(mone) # Since we maximize, we want to have the opposite gradient.
    D.optim.step()
    
    print("Loss: ", loss.item()," WD: ", WD.item())