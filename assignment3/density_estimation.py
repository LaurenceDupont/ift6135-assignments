#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""

from __future__ import print_function
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from samplers import distribution4

# Useful functions for plotting
f = lambda x: torch.tanh(x * 2 + 1) + x * 0.75
d = lambda x: (1 - torch.tanh(x * 2 + 1) ** 2) * 2 + 0.75
N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)


def plot_discriminator():
    '''
    Plot the output of your trained discriminator and the estimated density contrasted with the true density
    '''
    # plot the output of your trained discriminator
    xx = np.linspace(-5, 5, 1000)
    r = xx  # evaluate xx using your discriminator; replace xx with the output
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(xx, r)  # Not sure what this graph is
    plt.title(r'$D(x)$')

    # plot the estimated density contrasted with the true density
    estimate = N(torch.from_numpy(xx)).numpy()  # estimate the density of distribution4 (on xx) using the discriminator;
    plt.subplot(1, 2, 2)
    plt.plot(xx, estimate)
    plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
    plt.legend(['Estimated', 'True'])
    plt.title('Estimated vs True')
    plt.savefig('Plots_Q1/Estimated_exact_D4')


def plot_p0_p1():
    '''
    Plotting function
    '''
    plt.figure()
    # empirical
    xx = torch.randn(10000)
    plt.hist(f(xx), 100, alpha=0.5, density=1)
    plt.hist(xx, 100, alpha=0.5, density=1)
    plt.xlim(-5, 5)
    # exact
    xx = np.linspace(-5, 5, 1000)
    N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)
    plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
    plt.plot(xx, N(xx))
    plt.savefig('Plots_Q1/Empirical_Exact.png')


def jsd(p, q):
    '''
    Implementation of pairwise Jensen Shannon Divergence
    :param p: sequence Distribution p
    :param q: sequence Distribution q
    :return: float The calculated entropy
    '''
    p, q = np.asarray(p), np.asarray(q)
    p, q = p / p.sum(), q / q.sum()  # Normalize the distribution p
    r = (p + q) / 2.
    return np.sum(p * np.log(p / r), axis=0) / 2. + np.sum(q * np.log(q / r), axis=0) / 2. + np.log(2)

############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4


class MLP_Discriminator(nn.Module):
    def __init__(self):
        super(MLP_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear()