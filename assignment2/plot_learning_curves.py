#!/bin/python
# coding: utf-8

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

##############################################################################
#
# ARG PARSING
#
##############################################################################

parser = argparse.ArgumentParser(description='Plot learning curves')

parser.add_argument('--exp_dir', type=str, default='',
                    help='path to the problem4 directory, which contains the saved models')

args = parser.parse_args()

##############################################################################
#
# PLOT LEARNING CURVES
#
##############################################################################

def load_data(filename):
    return np.load(filename)[()]

def plot_epoch_learning_curves(plot_filename, plot_data):
    fig = plt.figure()
    
    for data in plot_data:
        plt.plot(np.arange(0,len(data['ppls']),1), data['ppls'], label=data['label'])

    plt.title("Perplexity by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()
    print()
    print(plot_filename)
    plt.show()
    
    fig.savefig(plot_filename)
    
# Experiments learning curves
def plot_train_val_learning_curves(save_dir, train_ppls, val_ppls):
    plot_filename = save_dir + 'epoch_plot.png'
    
    plot_data = [{'ppls': train_ppls, 'label': 'train'}, {'ppls': val_ppls, 'label': 'validation'}]
    
    plot_epoch_learning_curves(plot_filename, plot_data)

# Models and optimizers learning curves
def plot_val_learning_curves(save_dir, model_or_optimizer, val_ppls_list):
    plot_data = []
    
    for index, val_ppls in enumerate(val_ppls_list):
        plot_data.append({'ppls': val_ppls, 'label': index})

    plot_filename = save_dir + model_or_optimizer + '_epoch_plot.png'
    
    plot_epoch_learning_curves(plot_filename, plot_data)


DATA_FILENAME = 'learning_curves.npy'

# Model learning curves
RNN_val_ppls = []
GRU_val_ppls = []
TRANSFORMER_val_ppls = []

# Optimizer learning curves
SGD_LR_SCHEDULE_val_ppls = []
SGD_val_ppls = []
ADAM_val_ppls = []

for filename in glob.iglob(args.exp_dir + '/**', recursive=True):
    if os.path.isfile(filename):
        if filename.endswith(DATA_FILENAME):
            data = load_data(filename)
            train_ppls = data['train_ppls']
            val_ppls = data['val_ppls']
            
            if 'model=RNN' in filename:
                RNN_val_ppls.append(val_ppls)
            elif 'model=GRU' in filename:
                GRU_val_ppls.append(val_ppls)
            elif 'model=TRANSFORMER' in filename:
                TRANSFORMER_val_ppls.append(val_ppls)
                
            if 'optimizer=SGD_LR_SCHEDULE' in filename:
                SGD_LR_SCHEDULE_val_ppls.append(val_ppls)    
            elif 'optimizer=SGD' in filename:
                SGD_val_ppls.append(val_ppls)
            elif 'optimizer=ADAM' in filename:
                ADAM_val_ppls.append(val_ppls)
                
            save_dir = filename[:-len(DATA_FILENAME)]
                
            plot_train_val_learning_curves(save_dir, train_ppls, val_ppls)


save_dir = args.exp_dir + '/plots/'

plot_val_learning_curves(save_dir, 'RNN', RNN_val_ppls)
plot_val_learning_curves(save_dir, 'GRU', GRU_val_ppls)
plot_val_learning_curves(save_dir, 'TRANSFORMER', TRANSFORMER_val_ppls)

plot_val_learning_curves(save_dir, 'SGD_LR_SCHEDULE', SGD_LR_SCHEDULE_val_ppls)
plot_val_learning_curves(save_dir, 'SGD', SGD_val_ppls)
plot_val_learning_curves(save_dir, 'ADAM', ADAM_val_ppls)