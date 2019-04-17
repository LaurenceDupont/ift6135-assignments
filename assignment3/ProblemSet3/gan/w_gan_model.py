#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import torch
from torch import nn
from torch.optim import Adam
from torch import autograd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%

# inspired from our homework 1 and https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) # TODO: Potentially change to layer norm? 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        
        self.downsample = None
        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Reduce the sample size so the residual matches size
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class W_BaseDiscriminator(nn.Module):
    def __init__(self, Lambda = 11):
        super(W_BaseDiscriminator, self).__init__()
        self.Lambda = Lambda # Gradient penalty lambda hyperparameter

    def gradient_penalty(self, x_real, x_fake):
        alpha = torch.FloatTensor(x_real.size()).uniform_(0, 1).to(DEVICE)

        z = alpha * x_real.detach() + (1 - alpha) * x_fake.detach() # Detach from other forwards
        z.requires_grad_(True)

        T_z = self.forward(z) # T(z)
        
        grads = autograd.grad(outputs=T_z, inputs=z, # outputs w.r.t. the inputs
            grad_outputs=torch.ones(T_z.size()).to(DEVICE), # Keep all grads
            create_graph=True
        )[0]

        grad_penalty = ((grads.norm(2) -1) ** 2).mean() * self.Lambda
        return grad_penalty

    def loss_function(self, x_real, x_fake, forward_real, forward_fake):
        
        forward_real = forward_real.mean() # Mean over batch examples
        forward_fake = forward_fake.mean() # Mean over batch examples

        gradient_penalty = self.gradient_penalty(x_real, x_fake)

        d_loss = forward_real - forward_fake - gradient_penalty
        WD = forward_real - forward_fake
        return d_loss, WD


class W_NN_Discriminator(W_BaseDiscriminator):
    def __init__(self, input_size=512, hidden_size=512, lr=1e-3, Lambda=10):
        super(W_NN_Discriminator, self).__init__(Lambda=Lambda)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.discriminator = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, 1),
        )
        self.optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    def forward(self, x):
        x = x.view(x.shape[0], -1) # squeeze the H,W and chanels 
        y = self.discriminator(x)
        return y

# inspired from https://github.com/jalola/improved-wgan-pytorch
class W_CNN_Discriminator(W_BaseDiscriminator):
    def __init__(self, isBlackAndWhite=False, feature_dim=16, lr=1e-4, Lambda = 11):
        super(W_CNN_Discriminator, self).__init__(Lambda=Lambda)

        initial_size = 3 if not isBlackAndWhite else 1
        self.feature_dim = feature_dim
        self.initConv = conv3x3(initial_size, self.feature_dim)

        self.blockChain = nn.Sequential(
            BasicBlock(self.feature_dim, 2*self.feature_dim),
            BasicBlock(2*self.feature_dim, 4*self.feature_dim),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4*self.feature_dim, 1)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
    

    def forward(self, x):
        x = self.initConv(x)
        x = self.blockChain(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y



class W_Generator(nn.Module):
    DIMENSION_H = 256

    def __init__(self, latent_var_nb=100, lr=1e-4):
        super(W_Generator, self).__init__()

        self.decoder_fc = nn.Linear(latent_var_nb, self.DIMENSION_H)
        
        self.generator = nn.Sequential(
            nn.ELU(),
            
            nn.Conv2d(256, 64, 5, padding=4),
            nn.ELU(),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=2),
            nn.ELU(),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=2),   
            nn.ELU(),
            
            nn.Conv2d(16, 1, 3, padding=2) # Black and white...
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        x = self.decoder_fc(x).unsqueeze(2).unsqueeze(3)
        x = self.generator(x)
        return x