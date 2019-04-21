import random
import math
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import assignment3.samplers as samplers
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

# Code inspired from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

# 1.1 JSD Estimator

class Discriminator(nn.Module):
    """
    Discriminator for the Jensen-Shannon Estimator
    """
    def __init__(self, input_size=2, hidden_size=None, output_size=1):
        super(Discriminator, self).__init__()

        if hidden_size is None:
            self.hidden_size = []
        else:
            self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        mlp = []
        input_size_ = input_size

        for i in range(len(self.hidden_size)):
            mlp.append(nn.Linear(input_size_, self.hidden_size[i]))
            mlp.append(nn.ReLU())
            input_size_ = self.hidden_size[i]
        mlp.append(nn.Linear(self.hidden_size[-1], self.output_size))
        mlp.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.mlp(x)
        return validity


class Generator(nn.Module):
    def __init__(self, input_size=1, hidden_size=None, output_size=512):
        super(Generator, self).__init__()
        if hidden_size is None:
            self.hidden_size = []
        else:
            self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        mlp = []
        input_size_ = input_size

        for i in range(len(self.hidden_size)):
            mlp.append(nn.Linear(input_size_, self.hidden_size[i]))
            mlp.append(nn.ReLU())
            input_size_ = self.hidden_size[i]
        mlp.append(nn.Linear(self.hidden_size[-1], self.output_size))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, latent_variable):
        gen_sample = self.mlp(latent_variable)
        return gen_sample

def objectiveJS(true_validity, gen_validity, device='cuda:0'):
    """
    Objective to maximize (we're minimizing it's negative)
    :param output:
    :param target:
    :return:
    """
    # We add a minus in front of the objective so that we can minimize it w/ grad descent
    # bce_loss = nn.BCELoss(size_average=True) We could but won't use that objective
    D_x = torch.mean(torch.log(true_validity))
    D_y = torch.mean(torch.log(1.0-gen_validity))
    objective = -1.0*math.log(2) - 0.5*(D_x+D_y)
    return objective

def JensenShannonEstimator(true_distrib, gen_distrib, epochs=20):
    """
    JensenShannonEstimator
    :param true_distrib:
    :param gen_distrib:
    :param lr:
    :param epochs:
    :return:
    """
    discriminator = Discriminator(input_size=2, hidden_size=[50, 100, 50])
    discriminator = discriminator.to('cuda:0')
    optimizer = SGD(discriminator.parameters(), lr=0.1)
    discriminator.train()

    for epoch in range(epochs):
        true_Tensor = torch.Tensor(next(true_distrib)).to('cuda:0')
        gen_Tensor = torch.Tensor(next(gen_distrib)).to('cuda:0')
        true_Tensor.requires_grad = True
        gen_Tensor.requires_grad = True
        optimizer.zero_grad()
        # Forward and loss
        true_validity = discriminator(true_Tensor)
        gen_validity = discriminator(gen_Tensor)
        objective = objectiveJS(true_validity, gen_validity)
        # Step
        objective.backward()
        optimizer.step()

    # objective = np.mean(np.array(obj))
    return -1.0*objective

# 1.2 Wasserstein

class Critic(nn.Module):
    def __init__(self, input_size=2, hidden_size=None, output_size=1):
        super(Critic, self).__init__()

        if hidden_size is None:
            self.hidden_size = []
        else:
            self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        mlp = []
        input_size_ = input_size

        for i in range(len(self.hidden_size)):
            mlp.append(nn.Linear(input_size_, self.hidden_size[i]))
            mlp.append(nn.ReLU())
            input_size_ = self.hidden_size[i]
        mlp.append(nn.Linear(self.hidden_size[-1], self.output_size))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        score = self.mlp(x)
        return score


def objectiveW(critic_true, critic_gen, critic_gradpen, lambda_=10.0, device='cuda:0'):
    # We add a negative to the objective to be able to minimize it instead of maximizing it instead
    critic_true = torch.mean(critic_true)
    critic_gen = torch.mean(critic_gen)
    critic_gradpen = torch.pow(torch.mean(torch.norm(critic_gradpen, dim=1)-1.0),2)
    objective = -1.0*critic_true + critic_gen + lambda_*critic_gradpen
    w_d = critic_true - critic_gen
    return objective, w_d

def WassersteinEstimator(true_distrib, gen_distrib, epochs=20):
    """
    Wasserstein distance estimator
    :param true_distrib: ndarray # Already optimal batch size
    :param gen_distrib: ndarray # Already optimal batch size
    :param lr:
    :param epochs:
    :return:
    """
    # Creating dataset instances
    critic = Critic(input_size=2, hidden_size=[100, 200, 100])
    critic = critic.to('cuda:0')
    optimizer = SGD(critic.parameters(), lr=0.001)
    objective = 0
    w_d = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        true_tensor = torch.Tensor(next(true_distrib)).to('cuda:0')
        gen_tensor = torch.Tensor(next(gen_distrib)).to('cuda:0')
        a = torch.Tensor(np.array([random.uniform(0, 1) for _ in range(512)])[:, np.newaxis]).to('cuda:0')
        true_tensorTemp = true_tensor
        gen_tensorTemp = gen_tensor
        z = torch.mul(a, (true_tensorTemp.detach())) + torch.mul((torch.cuda.FloatTensor(1.0 - a)),(gen_tensorTemp.detach()))
        z.requires_grad = True
        true_pred = critic(true_tensor)
        gen_pred = critic(gen_tensor)
        z_pred = critic(z)
        grad = torch.autograd.grad(torch.mean(z_pred), z)[0]
        objective, w_d = objectiveW(true_pred, gen_pred, grad, lambda_=40)
        objective.backward()
        optimizer.step()

    return -1.0*objective, w_d



if __name__ == "__main__":

    # 1.3
    # Model

    p = iter(samplers.distribution1(0))
    phis = np.arange(-1, 1.1, 0.1)

    JS = []
    W = []

    for phi in phis:
        q = iter(samplers.distribution1(phi))
        JS.append(JensenShannonEstimator(p, q, epochs=1000))
        W.append(WassersteinEstimator(p, q, epochs=400)[1])

    JS = np.array(JS)
    W = np.array(W)

    plt.figure()
    plt.title("Estimates for Jensen-Shannon and Wasserstein distance")
    plt.plot(phis, JS, 'r', label="JS")
    plt.plot(phis, W, 'b', label="W")
    plt.legend()
    plt.show()