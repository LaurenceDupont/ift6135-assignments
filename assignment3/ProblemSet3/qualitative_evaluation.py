#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision.utils import save_image
import numpy as np

from gan.w_gan_model import W_Online3_Generator
from vae.vae_svhn import VAE

cuda = torch.cuda.is_available()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIMENSION_Z = 100


def load_gan_generator(model_path):
    model = W_Online3_Generator()
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    return model, model


def load_vae_generator(model_path):
    model = VAE()
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    return model, model.decode


def random_z():
    z = torch.randn(1, DIMENSION_Z)
    if cuda:
        z = z.cuda()
    return z


def sample(generator, samples_count, parent_directory):
    for sample_index in range(samples_count):
        z = random_z()
            
        sample = generator(z)
        save_image(sample[0].cpu(), f"{parent_directory}/samples/sample_{str(sample_index)}.png")
        
        
# Check if the model has learned a disentangled representation in the latent space
# - Sample a random z from the prior distribution
# - Make small perturbations to the sample z for each dimension
def z_perturbations(generator, parent_directory):
    z = random_z()
        
    sample = generator(z)
    save_image(sample.cpu(), f"{parent_directory}/z_perturbations/zz.png")
    
    for epsilon in np.arange(0, 12, 2):
    
        for z_index in range(DIMENSION_Z):
            z[0][z_index] = z[0][z_index] + epsilon
            sample = generator(z)
            save_image(sample.cpu(), f"{parent_directory}/z_perturbations/z_{str(z_index)}_{str(epsilon)}.png")
            z[0][z_index] = z[0][z_index] - epsilon
        

# Compare between interpolating in the data space and in the latent space
def interpolation(generator, parent_directory):
    a = torch.Tensor(np.arange(0, 1.1, 0.1))
    
    z0 = random_z()
    z1 = random_z()
    
    if cuda:
        z0 = z0.cuda()
        z1 = z1.cuda()
    
    x0 = generator(z0)
    x1 = generator(z1)

    for index, a_ in enumerate(a):
        z_prime = a_*z0 + (1.0-a_)*z1
        if cuda:
            z_prime = z_prime.cuda()
        x_prime = generator(z_prime)
        save_image(x_prime.unsqueeze(0).cpu(), f"{parent_directory}/interpolation/x_prime_{str(index)}.png")
        
        x_hat = a_*x0 + (1.0-a_)*x1
        save_image(x_hat.unsqueeze(0).cpu(), f"{parent_directory}/interpolation/x_hat_{str(index)}.png")
        
        
def qualitative_evaluation(model_name, model, generator):
    parent_directory=f"{model_name}/evaluation"
    
    with torch.no_grad():
        model.eval()
        
        sample(generator, samples_count=1000, parent_directory=parent_directory)
        z_perturbations(generator, parent_directory=parent_directory)
        interpolation(generator, parent_directory=parent_directory)
        

gan_model, gan_generator = load_gan_generator('gan/saved_models/w-gan-gp_Online3_Generator_61500.pt')
qualitative_evaluation('gan', gan_model, gan_generator)

vae_model, vae_generator = load_vae_generator('vae/saved_models/vae_svhn.pt')
qualitative_evaluation('vae', vae_model, vae_generator)