#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import sys
#sys.path.append('C:/Users/Game/AI/ift6135-assignments/assignment3/ProblemSet3/gan') # To solve import on Jupiter server

from timeit import default_timer as timer

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torchvision.utils import save_image
from w_gan_model import *
import w_gan_model

#%%
######################
## Const declaration
######################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GanTrainner():
    def __init__(self, LATENT_VAR_NB=100, NB_ITR_GENERATOR=1, NB_ITR_CRITIC=3):
        self.LATENT_VAR_NB = LATENT_VAR_NB
        self.NB_ITR_GENERATOR = NB_ITR_GENERATOR
        self.NB_ITR_CRITIC = NB_ITR_CRITIC
        self.discriminator_input_size = 784

    def rand_noise(self, batch_size):
        e= torch.FloatTensor(batch_size, self.LATENT_VAR_NB).normal_(0, 1).to(DEVICE)
        #e = torch.randn(batch_size, self.LATENT_VAR_NB).to(DEVICE)
        #e = torch.randn(MINI_BATCH_SIZE, IMAGE_SIZE*2)
        return e

    def gen_img(self, Gen_model, itr, noise=None):
        with torch.no_grad():
            noise = self.rand_noise(8)
            x = Gen_model(noise)
        # Source for reconstructing the image: https://github.com/pytorch/examples/blob/master/vae/main.py
        n = min(x.size(0), 8)
        comparison = x[:n]
        save_image(comparison.cpu(), './gan_results/reconstruction_' + str(itr) + '.png', nrow=n)


    def generate_image_from_saved_model(self, model_name, class_name):
        Generator = getattr(w_gan_model, class_name)()
        Generator.load_state_dict(torch.load("./saved_models/" + model_name))
        Generator.to(DEVICE)
        Generator.eval()
        
        self.gen_img(Generator, -1)

    ######################
    ## Train
    ## inspired from https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
    ######################
    def start(self, train, BATCH_SIZE, NB_TRAIN_LOOPS=10001, mnist=False, save_name="w_gan_generator"):
        Crit = W_NN_Discriminator(input_size=self.discriminator_input_size).to(DEVICE)
        #Crit = W_CNN_Discriminator(feature_dim=32).to(DEVICE)
        #Crit = W_Online2_Discriminator().to(DEVICE)
        #Crit = W_Online3_Discriminator().to(DEVICE)

        #Gen = W_Generator(latent_var_nb=self.LATENT_VAR_NB) if not mnist else W_Generator_monocrome_28(latent_var_nb=self.LATENT_VAR_NB) 
        Gen = W_Online_Generator(latent_var_nb=self.LATENT_VAR_NB)
        #Gen = W_Online2_Generator(latent_var_nb=self.LATENT_VAR_NB)
        #Gen = W_Online3_Generator()

        Gen = Gen.to(DEVICE)
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
            for i in range(self.NB_ITR_GENERATOR):
                Gen.optim.zero_grad()

                noise = self.rand_noise(BATCH_SIZE)
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

            for i in range(self.NB_ITR_CRITIC):
                
                Crit.optim.zero_grad()

                # Get new Real data each time
                batch = next(data_iter, None)
                if batch is None:
                    data_iter = iter(train)
                    batch = data_iter.next()
                real_data = batch[0].to(DEVICE)


                # Generate Fake Data and freeze the Gen
                noise = self.rand_noise(real_data.shape[0])
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
                self.gen_img(Gen, i_val)
            if (i_val % 500 == 0):
                torch.save(Gen.state_dict(), "./saved_models/" + save_name+ "_" + str(i_val) + ".pt")
            i_val += 1


        return Gen, Crit
        
