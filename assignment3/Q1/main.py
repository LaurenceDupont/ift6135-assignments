import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from density_estimation import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = MLP_Discriminator().to(DEVICE)
batch_size = 512
lr = 0.001
max_epoch = 20
criterion = nn.BCELoss() # METTRE LA BONNE LOSS
D_opt = torch.optim.Adam(D.parameters(), lr=lr)
D_labels = torch.ones(batch_size, 1).to(DEVICE)  # Discriminator Label to real
D_fakes = torch.zeros(batch_size, 1).to(DEVICE)  # Discriminator Label to fake

for epoch in range(max_epoch):
    x = f(np.linspace(-5, 5, batch_size))
    x_output = D(x)
    D_x_loss = criterion(x_output, D_labels)

    z = distribution4(batch_size)
    z_outputs = D(z)
    D_z_loss = criterion(z_outputs, D_fakes)
    D_loss = D_x_loss + D_z_loss

    D.zero_grad()
    D_loss.backward()
    D_opt.step()