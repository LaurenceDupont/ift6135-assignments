
import os
import numpy as np

import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils

from train import GanTrainner

def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=transforms.ToTensor()
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = data_utils.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader

if __name__ == "__main__":
    BATCH_SIZE = 128
    train, valid, test = get_data_loader("svhn", BATCH_SIZE)

    #######################
    ## Train
    #######################
    ganTrainner = GanTrainner(NB_ITR_CRITIC=5, NB_ITR_GENERATOR=1)
    ganTrainner.discriminator_input_size = 3072 # 3*32*32
    Gen, Crit = ganTrainner.start(train, BATCH_SIZE, NB_TRAIN_LOOPS=200001, save_name="w-gan-gp_Online3_Generator")

    #ganTrainner.generate_image_from_saved_model("w-gan-gp_Online3_Generator_61500.pt", "W_Online3_Generator")