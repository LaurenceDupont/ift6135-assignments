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

def plot_epoch_learning_curves(save_dir, exp_name, plot_data):
    plot_filename = save_dir + exp_name + '-epochs.png'
    
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
    
def plot_wall_clock_time_learning_curves(save_dir, exp_name, plot_data):
    plot_filename = save_dir + exp_name + '-time.png'
    
    fig = plt.figure()
    
    for data in plot_data:
        plt.plot(data['times'], data['ppls'], label=data['label'])

    plt.title("Perplexity by Wall-Clock Time")
    plt.xlabel("Wall-Clock Time (s)")
    plt.ylabel("Perplexity")
    plt.legend()
    print()
    print(plot_filename)
    plt.show()
    
    fig.savefig(plot_filename)
    
# Experiments learning curves
def plot_train_val_learning_curves(save_dir, exp_name, train_ppls, val_ppls, times):
    plot_data = [{'times': times, 'ppls': train_ppls, 'label': 'train'}, {'times': times, 'ppls': val_ppls, 'label': 'validation'}]
    
    plot_epoch_learning_curves(save_dir, exp_name, plot_data)
    plot_wall_clock_time_learning_curves(save_dir, exp_name, plot_data)

# Models and optimizers learning curves
def plot_val_learning_curves(save_dir, exp_name, plot_data):
    plot_epoch_learning_curves(save_dir, exp_name, plot_data)
    plot_wall_clock_time_learning_curves(save_dir, exp_name, plot_data)


DATA_FILENAME = 'learning_curves.npy'

# Model learning curves
rnn_plot_data = []
gru_plot_data = []
transformer_plot_data = []

# Optimizer learning curves
sgd_lr_plot_data = []
sgd_plot_data = []
adam_plot_data = []

save_dir = args.exp_dir + '/plots/'

for filename in glob.iglob(args.exp_dir + '/**', recursive=True):
    if os.path.isfile(filename):
        if filename.endswith(DATA_FILENAME):
            exp_name = filename.split('/')[-2]
            
            data = load_data(filename)
            train_ppls = data['train_ppls']
            val_ppls = data['val_ppls']
            times = data['times']
            
            plot_data = {'times': times, 'ppls': val_ppls, 'label': exp_name}
            
            if 'rnn' in exp_name:
                rnn_plot_data.append(plot_data)
            elif 'gru' in exp_name:
                gru_plot_data.append(plot_data)
            elif 'transformer' in exp_name:
                transformer_plot_data.append(plot_data)
            if 'sgd-lr' in exp_name:
                sgd_lr_plot_data.append(plot_data)
            elif 'sgd' in exp_name:
                sgd_plot_data.append(plot_data)
            elif 'adam' in exp_name:
                adam_plot_data.append(plot_data)
                
            plot_train_val_learning_curves(save_dir, exp_name, train_ppls, val_ppls, times)

plot_val_learning_curves(save_dir, 'rnn-summary', rnn_plot_data)
plot_val_learning_curves(save_dir, 'gru-summary', gru_plot_data)
plot_val_learning_curves(save_dir, 'transformer-summary', transformer_plot_data)

plot_val_learning_curves(save_dir, 'sgd-lr-summary', sgd_lr_plot_data)
plot_val_learning_curves(save_dir, 'sgd-summary', sgd_plot_data)
plot_val_learning_curves(save_dir, 'adam-summary', adam_plot_data)