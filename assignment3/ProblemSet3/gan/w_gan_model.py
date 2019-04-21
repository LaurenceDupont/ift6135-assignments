#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import torch
from torch import nn
from torch.optim import Adam
from torch import autograd
import torch.nn.functional as F
import numpy as np

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
        #self.bn1 = nn.BatchNorm2d(planes) # TODO: Potentially change to layer norm? 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
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
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        
        # Reduce the sample size so the residual matches size
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class W_BaseDiscriminator(nn.Module):
    def __init__(self, Lambda = 10):
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

#################################################
## inspired from https://github.com/jalola/improved-wgan-pytorch
#################################################
class W_CNN_Discriminator(W_BaseDiscriminator):
    def __init__(self, isBlackAndWhite=False, feature_dim=16, lr=1e-3, Lambda = 11):
        super(W_CNN_Discriminator, self).__init__(Lambda=Lambda)

        initial_size = 3 if not isBlackAndWhite else 1
        self.feature_dim = feature_dim
        self.initConv = conv3x3(initial_size, self.feature_dim)

        self.blockChain = nn.Sequential(
            BasicBlock(self.feature_dim, 2*self.feature_dim),
            #BasicBlock(2*self.feature_dim, 4*self.feature_dim),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2*self.feature_dim, 1)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
    

    def forward(self, x):
        x = self.initConv(x)
        x = self.blockChain(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y

#################################################
## inspired from https://github.com/shariqiqbal2810/WGAN-GP-PyTorch/blob/master/code/utils.py
#################################################
class W_Online2_Discriminator(W_BaseDiscriminator):
    def __init__(self, isBlackAndWhite=False, dim_factor=64, lr=1e-4, Lambda = 10):
        super(W_Online2_Discriminator, self).__init__(Lambda=Lambda)

        self.conv1 = nn.Conv2d(3, dim_factor, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_factor, 2 * dim_factor, 5,
                               stride=2, padding=2)
        self.conv3 = nn.Conv2d(2 * dim_factor, 4 * dim_factor, 5,
                               stride=2, padding=2)
        self.linear = nn.Linear(4096, 1)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        H1 = F.leaky_relu(self.conv1(X), negative_slope=0.2)
        H2 = F.leaky_relu(self.conv2(H1), negative_slope=0.2)
        H3 = F.leaky_relu(self.conv3(H2), negative_slope=0.2)
        H3_resh = H3.view(H3.size(0), -1)  # reshape for linear layer
        disc_out = self.linear(H3_resh)
        return disc_out

#############################
## from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
#############################
class W_Online3_Discriminator(W_BaseDiscriminator):
    def __init__(self, DIM=128, lr=1e-4):
        super(W_Online3_Discriminator, self).__init__()
        self.DIM = DIM
        self.main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.linear = nn.Linear(4*4*4*DIM, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output

class W_Online3_Generator(nn.Module):
    def __init__(self, latent_var_nb=100, DIM=128, lr=1e-4):
        super(W_Online3_Generator, self).__init__()
        self.DIM = DIM
        self.preprocess = nn.Sequential(
            nn.Linear(latent_var_nb, 4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.sig = nn.Sigmoid()

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sig(output)
        return output.view(-1, 3, 32, 32)

#------------------------------------------------------------------#


#############################
## from https://github.com/shariqiqbal2810/WGAN-GP-PyTorch/blob/master/code/utils.py
#############################
class W_Online2_Generator(nn.Module):
    DIMENSION_H = 256
    def __init__(self, latent_var_nb=100, lr=1e-4, dim_factor=64):
        super(W_Online2_Generator, self).__init__()
        self.H_init = int(32 / 2**3)  # divide by 2^3 bc 3 deconvs with stride 0.5
        self.W_init = int(32 / 2**3)  # divide by 2^3 bc 3 deconvs with stride 0.5

        self.linear = nn.Linear(latent_var_nb,
                                4 * dim_factor * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(4 * dim_factor, 2 * dim_factor,
                                          4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2 * dim_factor, dim_factor,
                                          4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(dim_factor, 3,
                                          4, stride=2, padding=1)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        H1 = F.relu(self.linear(X))
        H1_resh = H1.view(H1.size(0), -1, 4, 4)
        H2 = F.relu(self.deconv1(H1_resh))
        H3 = F.relu(self.deconv2(H2))
        img_out = F.sigmoid(self.deconv3(H3))
        return img_out

#############################
## from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
#############################
class W_Online_Generator(nn.Module):
    DIMENSION_H = 256
    img_shape = (3, 32, 32)

    def __init__(self, latent_var_nb=100, lr=1e-4):
        super(W_Online_Generator, self).__init__()

        self.decoder_fc = nn.Linear(latent_var_nb, self.DIMENSION_H)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_var_nb, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Sigmoid()
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        img = self.model(x)
        img = img.view(img.shape[0], *self.img_shape)
        return img

#############################
## Outputs a 32x32 3-chanel image format
#############################
class W_Generator(nn.Module):
    DIMENSION_H = 256

    def __init__(self, latent_var_nb=100, lr=1e-4):
        super(W_Generator, self).__init__()

        self.decoder_fc = nn.Linear(latent_var_nb, self.DIMENSION_H)

        self.generator = nn.Sequential(
            nn.ELU(),
            
            nn.Conv2d(256, 128, 4, padding=3),
            nn.ELU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 4, padding=1),
            nn.ELU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=2),
            nn.ELU(),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),   
            nn.ELU(),
            
            nn.Conv2d(16, 3, 3, padding=1),
        )
        #self.generator = nn.Sequential(
        #    nn.ELU(),
        #    
        #    nn.Conv2d(256, 64, 5, padding=4),
        #    nn.ELU(),
        #    
        #    nn.UpsamplingBilinear2d(scale_factor=2),
        #    nn.Conv2d(64, 32, 3, padding=2),
        #    nn.ELU(),
        #    
        #    nn.UpsamplingBilinear2d(scale_factor=2),
        #    nn.Conv2d(32, 16, 3, padding=2),   
        #    nn.ELU(),
        #    
        #    nn.Conv2d(16, 3, 3, padding=4)
        #)
        self.sig = nn.Sigmoid()

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        x = self.decoder_fc(x).unsqueeze(2).unsqueeze(3)
        x = self.generator(x)
        return self.sig(x)

#############################
## Outputs a 28x28 1-chanel image format
#############################
class W_Generator_monocrome_28(nn.Module):
    DIMENSION_H = 256

    def __init__(self, latent_var_nb=100, lr=1e-4):
        super(W_Generator_monocrome_28, self).__init__()

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