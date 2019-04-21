
import torch.utils.data as data_utils
import os
import numpy as np

import torch

from train import GanTrainner

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
train, valid, test = get_data_loader("../../datasets/MNIST_static", MINI_BATCH_SIZE)

#%%
#######################
## Train
#######################
ganTrainner = GanTrainner()
Gen, Crit = ganTrainner.start(train, MINI_BATCH_SIZE, NB_TRAIN_LOOPS=100001)