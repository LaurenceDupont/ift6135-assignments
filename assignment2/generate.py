#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Implementation reference: https://github.com/ceshine/examples/blob/master/word_language_model/generate.py

import argparse

import torch
import collections
import os

from models import RNN, GRU

device = None
# Use the GPU if you have one
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")


##############################################################################
#
# HELPER FUNCTIONS FROM PTB-LM.PY
#
##############################################################################

def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


##############################################################################
#
# HELPER FUNCTIONS
#
##############################################################################

def create_model(model_name, batch_size):
    if model_name == 'RNN':
        return RNN(emb_size=200, hidden_size=1500, 
                seq_len=35, batch_size=batch_size, 
                vocab_size=vocab_size, num_layers=2, 
                dp_keep_prob=0.35)
        
    if model_name == 'GRU':
        return GRU(emb_size=200, hidden_size=1500, 
                seq_len=35, batch_size=batch_size, 
                vocab_size=vocab_size, num_layers=2, 
                dp_keep_prob=0.35)
        
    raise Exception('Unsupported model: ' + model_name)


def generate_sequences(model_name, state_dict, inputs, batch_size, generated_seq_len):
    model = create_model(model_name, batch_size).to(device)
    
    with open(state_dict, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    model.eval()
    
    hidden = model.init_hidden()
    hidden.to(device)
    
    samples = torch.t(model.generate(inputs, hidden, generated_seq_len))
    
    print(f'Model: {model_name}, seq_len: {generated_seq_len}')
    print_sequences(samples)
    print()
    
    
def print_sequences(samples):
    for batch_index in range(samples.size()[0]):
        sequence = []
        for seq_index in range(samples.size()[1]):
            id = samples[batch_index][seq_index].item()
            word = id_to_word[id]
            sequence.append(word)
        print(' '.join(sequence))


##############################################################################
#
# ARG PARSING
#
##############################################################################

parser = argparse.ArgumentParser(description='Language Generation')

parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--rnn_state_dict', type=str, default='./best_params.pt',
                    help='best params')
parser.add_argument('--gru_state_dict', type=str, default='./best_params.pt',
                    help='best params')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()


##############################################################################
#
# SEQUENCE GENERATION
#
##############################################################################

train_path = os.path.join(args.data, "ptb.train.txt")

rnn_state_dict = args.rnn_state_dict
gru_state_dict = args.gru_state_dict

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

word_to_id, id_to_word = _build_vocab(train_path)

vocab_size = len(word_to_id)
batch_size = 10
seq_len = 35

inputs = torch.randint(vocab_size, (1, batch_size), dtype=torch.long).to(device)[0]

generate_sequences('RNN', rnn_state_dict, inputs, batch_size, seq_len)
generate_sequences('GRU', gru_state_dict, inputs, batch_size, seq_len)
generate_sequences('RNN', rnn_state_dict, inputs, batch_size, seq_len * 2)
generate_sequences('GRU', gru_state_dict,  inputs, batch_size, seq_len * 2)