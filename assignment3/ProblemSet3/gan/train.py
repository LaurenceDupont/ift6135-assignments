#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
import os
import sys
sys.path.append('C:/Users/Game/AI/ift6135-assignments/assignment3/ProblemSet3/gan')

from timeit import default_timer as timer

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import torch.utils.data as data_utils
from torchvision.utils import save_image
from w_gan_model import *
#%%
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

######################
## Load Data
######################
MINI_BATCH_SIZE = 128
train, valid, test = get_data_loader("C:/Users/Game/AI/ift6135-assignments/assignment3/datasets/MNIST_static", MINI_BATCH_SIZE)

#%%
######################
## Const declaration
######################
NB_TRAIN_LOOPS=10001
NB_ITR_GENERATOR=1
NB_ITR_CRITIC=3
LATENT_VAR_NB=100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rand_noise(batch_size):
    #e= torch.FloatTensor((batch_size, LATENT_VAR_NB)).normal_(0, 1).to(DEVICE)
    e = torch.randn(batch_size, LATENT_VAR_NB).to(DEVICE)
    #e = torch.randn(MINI_BATCH_SIZE, IMAGE_SIZE*2)
    return e

def gen_img(Gen_model, itr, noise=None):
    with torch.no_grad():
        noise = rand_noise(8)
        x = Gen_model(noise)
    # Source for reconstructing the image: https://github.com/pytorch/examples/blob/master/vae/main.py
    n = min(x.size(0), 8)
    comparison = x[:n]
    save_image(comparison.cpu(), 'C:/Users/Game/AI/ift6135-assignments/assignment3/ProblemSet3/gan/gan_results/reconstruction_' + str(itr) + '.png', nrow=n)


######################
## Train
## inspired from https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
######################
def train_loop():
    Crit = W_NN_Discriminator(input_size=784).to(DEVICE)
    #Crit = W_CNN_Discriminator(isBlackAndWhite=True).to(DEVICE)
    Gen = W_Generator(latent_var_nb=LATENT_VAR_NB).to(DEVICE)
    mone = torch.FloatTensor([-1]).to(DEVICE)
    data_iter = iter(train)

    i_val = 0
    for i in range(NB_TRAIN_LOOPS):

        ## for batch_index, batch in enumerate(train): <--- We can't do this since we want to train the 
                                                #           Crit multiple times before the next iteration.
                                                #           Each time taking a new example for the Crit.
        
        starting_time = timer()

        ##############################
        ### Train Generator ###
        for param in Crit.parameters():
            param.requires_grad_(False)  # don't learn for the Discriminator

        gen_cost = None
        for i in range(NB_ITR_GENERATOR):
            Gen.optim.zero_grad()

            noise = rand_noise(MINI_BATCH_SIZE)
            noise.requires_grad_(True)

            fake_data = Gen(noise)
            gen_cost = Crit(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
        
        Gen.optim.step()



        ##############################
        ### Train Discriminator ###
        for param in Crit.parameters():
            param.requires_grad_(True) # Reset learning to ON

        for i in range(NB_ITR_CRITIC):
            
            Crit.optim.zero_grad()

            # Get new Real data each time
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(train)
                batch = data_iter.next()
            real_data = batch[0].to(DEVICE)


            # Generate Fake Data and freeze the Gen
            noise = rand_noise(real_data.shape[0])
            with torch.no_grad():   # freeze Gen, we train Crit only. we can't use "param.requires_grad_(False)" 
                noise_init = noise  # since we need the leaf on detach
            fake_data = Gen(noise_init).detach()

            # Train Critic
            f_x_real = Crit(real_data) 
            f_x_fake = Crit(fake_data)

            loss, WD = Crit.loss_function(real_data, fake_data, f_x_real, f_x_fake)

            loss.backward(mone) # Since we maximize, we want to have the opposite gradient.
            Crit.optim.step()

        ## Stop timer
        final_time = timer()
        if (i_val % 10 == 0):
            print("")
            print(f"Itr: {i_val}, Elapsed {final_time-starting_time}")
            print(f"Loss: {loss.item()}, WD: { WD.item()}")
        if (i_val % 100 == 0):
            gen_img(Gen, i_val)
        i_val += 1
        


train_loop()
