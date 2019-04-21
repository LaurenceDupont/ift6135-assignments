
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019
@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch
import matplotlib.pyplot as plt

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))
plt.show()

############### import the sampler ``samplers.distribution4''
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

=======
import math
import assignment3.samplers as samplers
from torch.optim.sgd import SGD
from assignment3.question1 import objectiveJS, Discriminator
from torch.utils.data.dataloader import DataLoader

def new_objective(f1_validity, f0_validity):
    D_x = torch.mean(torch.log(f1_validity))
    D_y = torch.mean(torch.log(1.0 - f0_validity))
    objective = -1.0*(D_x + D_y)
    return objective

def f1_density_estimator(x, discriminator, f0):
    """
    F1 density estimation function. Uses the question 5 procedure: f1(x) = f0(x) _dot_ D*(x)/(1-D*(x))
    :param x: point to evaluate
    :param discriminator: trained discriminator
    :param f0: density function
    :return: f1 density estimate for point
    """
    discriminator.eval()
    x_ = torch.Tensor(x)
    x_dl = DataLoader(x_, batch_size=512)
    output = torch.Tensor([]).to('cuda:0')
    for minibatch in x_dl:
        minibatch = minibatch.to('cuda:0')
        output = torch.cat((output, discriminator(minibatch)), dim=0)
    output = output.cpu().detach().numpy()
    f0_ = f0(x)
    numerator = (np.multiply(f0_,output.squeeze()))
    denominator = (1.0-output.squeeze())
    return np.divide(numerator,denominator)

p0_gen = iter(samplers.distribution3(batch_size=512))
p0 = next(p0_gen)

p1_gen = iter(samplers.distribution4(batch_size=512))
p1 = next(p1_gen)

epochs = 10000

discriminator = Discriminator(input_size=1, hidden_size=[100, 300, 150])
discriminator = discriminator.to('cuda:0')
optimizer = SGD(discriminator.parameters(), lr=0.001)
discriminator.train()


for epoch in range(epochs):
    p0_Tensor = torch.Tensor(next(p0_gen)).to('cuda:0')
    p1_Tensor = torch.Tensor(next(p1_gen)).to('cuda:0')
    p0_Tensor.requires_grad = True
    p1_Tensor.requires_grad = True
    optimizer.zero_grad()
    # Forward and loss
    f0_validity = discriminator(p0_Tensor)
    f1_validity = discriminator(p1_Tensor)
    objective = new_objective(f1_validity, f0_validity)
    # Step
    objective.backward()
    optimizer.step()

#print(-1.0*objective)

xx_tensor = torch.Tensor(xx)
xx_dl = DataLoader(xx_tensor, batch_size=512)
output = torch.Tensor([]).to('cuda:0')
for xx_ in xx_dl:
    xx_ = xx_.to('cuda:0')
    output = torch.cat((output, discriminator(xx_)), dim=0)
output = output.cpu()
output = output.detach().numpy()




############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density



r = output # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r.squeeze())
plt.title(r'$D(x)$')

estimate = f1_density_estimator(xx, discriminator, N) # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate

plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')
plt.show()

