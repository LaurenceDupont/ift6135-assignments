import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 

device = None
# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda") 
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The numvwe of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers. 
        # Your implementation should support any number of stacked hidden layers 
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules 
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
        # provided clones function (as opposed to a regular python list), in order 
        # for Pytorch to recognize these parameters as belonging to this nn.Module 
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.
        
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_size)
        
        W_x = []
        W_x.append(nn.Linear(emb_size, hidden_size, bias=False)) # (emb_size, hidden_size) for the first layer
        for layer_index in range(num_layers-1): # (hidden_size, hidden_size) for subsequent layers
            W_x.append(copy.deepcopy(nn.Linear(hidden_size, hidden_size, bias=False)))
        
        self.W_x = nn.ModuleList(W_x)
        
        self.W_h = clones(nn.Linear(hidden_size, hidden_size), num_layers)
        
        self.dropout = nn.Dropout(p=1-dp_keep_prob)
        self.tanh = nn.Tanh()
        
        self.W_y = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights_uniform()

    def init_weights_uniform(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
        # in the range [-k, k] where k is the square root of 1/hidden_size
        
        a = -0.1
        b = 0.1
        
        k = 1.0 / np.sqrt(self.hidden_size)
        
        self.embedding.weight = nn.init.uniform_(self.embedding.weight, a, b)
        
        for layer_index in range(self.num_layers):
            self.W_x[layer_index].weight = nn.init.uniform_(self.W_x[layer_index].weight, -k, k)
            self.W_h[layer_index].weight = nn.init.uniform_(self.W_h[layer_index].weight, -k, k)
            self.W_h[layer_index].bias = nn.init.uniform_(self.W_h[layer_index].bias, -k, k)
        
        self.W_y.weight = nn.init.uniform_(self.W_y.weight, a, b)
        self.W_y.bias.data.fill_(0)
        
        return

    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        hidden = nn.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size), requires_grad=True).to(device)
        return hidden # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        #return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device) # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the 
        # inner for loop should iterate over hidden layers of the stack. 
        # 
        # Within these for loops, use the parameter tensors and/or nn.modules you 
        # created in __init__ to compute the recurrent updates according to the 
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that 
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
        
        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the 
                  mini-batches in an epoch, except for the first, where the return 
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details, 
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """

        """
        From the forum:
        
        Only apply dropout "up the stack" not "forward through time"
        
        In short, the dropout is applied to the hidden activations before they are treated as input in the next layer of the stack.

        So (conceptually) you would have:
        - one copy of the hidden state at layer l-1 which has NO dropout (which is used to compute the hidden state at the NEXT timestep for that layer)
        - another copy WITH dropout, which is used to compute the hidden state of layer l on the SAME timestep
        
        Refer to figure 2 in the following paper: https://arxiv.org/pdf/1409.2329.pdf
        """
        
        seq_len = inputs.size()[0]
        
        embedded = self.embedding(inputs)
        
        hidden_without_dropout = []
        logits = []
        
        for timestep in range(seq_len):
            hidden_with_dropout = []
            hidden_without_dropout.append([])
            
            for layer_index in range(self.num_layers):
                input_W_x = self.dropout(embedded[timestep]) if layer_index == 0 else hidden_with_dropout[layer_index-1]
                input_W_h = hidden[layer_index] if timestep == 0 else hidden_without_dropout[timestep-1][layer_index]
                
                hidden_value = self.tanh(self.W_x[layer_index](input_W_x) + self.W_h[layer_index](input_W_h))
                
                hidden_without_dropout[timestep].append(hidden_value)
                hidden_with_dropout.append(self.dropout(hidden_value))
                
            hidden_with_dropout = torch.stack(hidden_with_dropout)
            
            logits_t = self.W_y(hidden_with_dropout[self.num_layers-1])
            logits.append(logits_t)
        
        logits = torch.stack(logits)
        
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden_with_dropout

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        # 
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution, 
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation 
        # function here in order to compute the parameters of the categorical 
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used 
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        
        return samples


# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for 
    GRU, not Vanilla RNN.
    """
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob


        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)
        self.emb_dropout = nn.Dropout(p=(1-self.dp_keep_prob))

        core = [GRU_unit(self.emb_size, self.hidden_size, self.batch_size, self.dp_keep_prob).to(device)]
        if self.num_layers>1:
            core.extend([GRU_unit(self.hidden_size, self.hidden_size, self.batch_size, self.dp_keep_prob).to(device) for i in range(num_layers-1)])

        self.core = nn.ModuleList(core)

        self.out = nn.Linear(hidden_size, vocab_size)

        self.init_weights_uniform()

    def init_weights_uniform(self):
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
        # in the range [-k, k] where k is the square root of 1/hidden_size
        
        a = -0.1
        b = 0.1
        k = 1.0 / np.sqrt(self.hidden_size)
        
        self.embedding.weight = nn.init.uniform_(self.embedding.weight, a, b)
        self.out.weight = nn.init.uniform_(self.out.weight, a, b)
        self.out.bias.data.fill_(0)
        
        for i in range(len(self.core)):
            self.core[i].init_weights(k=k)

        return None

    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        return torch.zeros((self.num_layers, self.batch_size, self.hidden_size), requires_grad=False).to(device)

    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        seq_len = inputs.size()[0]
        
        embedded = self.embedding(inputs)
        
        hidden_without_dropout = []
        logits = []
        
        for timestep in range(seq_len):
            hidden_with_dropout = []
            hidden_without_dropout.append([])
            
            for layer_index in range(self.num_layers):
                input_W_x = self.emb_dropout(embedded[timestep]) if layer_index == 0 else hidden_with_dropout[layer_index-1]
                input_W_h = hidden[layer_index] if timestep == 0 else hidden_without_dropout[timestep-1][layer_index]
                
                dropout_output, output = self.core[layer_index](input_W_x, input_W_h)
                
                hidden_without_dropout[timestep].append(output)
                hidden_with_dropout.append(dropout_output)
                
            hidden_with_dropout = torch.stack(hidden_with_dropout)
            
            logits_t = self.out(hidden_with_dropout[self.num_layers-1])
            logits.append(logits_t)
        
        logits = torch.stack(logits)
        
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden_with_dropout # torch.stack(hidden_without_dropout[-1])


    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        ## hidden_last_timestep = [] # No dropout
        ## hidden_current_timestep = [] # Dropout
        ## samples = []
## 
        ## emb_input = self.embedding(input)
## 
        ## for timestep in range(generated_seq_len):
        ##     emb_current_input = self.emb_dropout(emb_input[timestep])
        ##     hidden_last_timestep.append([])
        ##     for layer in range(self.num_layers):
        ##         input_W_x = emb_current_input if layer == 0 else hidden_current_timestep[layer-1]
        ##         input_W_h = hidden[layer] if timestep == 0 else hidden_last_timestep[timestep-1][layer]
## 
        ##         dropout_output, output = self.core[layer](input_W_x, input_W_h)
        ##         hidden_last_timestep[timestep].append(output)
        ##         hidden_current_timestep.append(dropout_output)
## 
        ##     softmax_output = F.softmax(self.out(hidden_current_timestep[-1]), dim=1)
        ##     pred = torch.argmax(softmax_output, dim=1)
        ##     samples.append(pred)
## 
        ## return samples

class GRU_unit(nn.Module):
    def __init__(self, emb_size, hidden_size, batch_size, dp_keep_prob):
        super(GRU_unit, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.dp_keep_prob = dp_keep_prob

        self.rt_input = nn.Linear(emb_size, hidden_size)
        self.rt_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rt_act = nn.Sigmoid()

        self.zt_input = nn.Linear(emb_size, hidden_size)
        self.zt_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.zt_act = nn.Sigmoid()

        self.htilde_input = nn.Linear(emb_size, hidden_size)
        self.htilde_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.htilde_act = nn.Tanh()

        self.ones = torch.ones(self.hidden_size, dtype=torch.float).to(device)

        #self.out = nn.Linear(hidden_size, hidden_size),
        self.dropout = nn.Dropout(p=(1-self.dp_keep_prob))

    def init_weights(self, k=None):
        if k is None:
            return

        self.rt_input.weight = nn.init.uniform_(self.rt_input.weight, -k, k)
        self.rt_input.bias.data.fill_(0)
        self.rt_hidden.weight = nn.init.uniform_(self.rt_hidden.weight, -k, k)
        #self.rt_hidden.bias.data.fill_(0)

        self.zt_input.weight = nn.init.uniform_(self.zt_input.weight, -k, k)
        self.zt_input.bias.data.fill_(0)
        self.zt_hidden.weight = nn.init.uniform_(self.zt_hidden.weight, -k, k)
        #self.zt_hidden.bias.data.fill_(0)

        self.htilde_input.weight = nn.init.uniform_(self.htilde_input.weight, -k, k)
        self.htilde_input.bias.data.fill_(0)
        self.htilde_hidden.weight = nn.init.uniform_(self.htilde_hidden.weight, -k, k)
        #self.htilde_hidden.bias.data.fill_(0)

    def forward(self, input, hidden):
        rt = self.rt_act(self.rt_input(input) + self.rt_hidden(hidden))
        zt = self.zt_act(self.zt_input(input) + self.zt_hidden(hidden))
        htilde = self.htilde_act(self.htilde_input(input) + self.htilde_hidden(torch.mul(hidden, rt)))
        # h = (self.ones - zt) * hidden + zt * htilde
        h = torch.mul((torch.ones((input.size()[0], self.hidden_size), dtype=torch.float).to(device).sub(zt)),hidden).add(torch.mul(zt,htilde))
        return self.dropout(h), h


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units 
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p=dropout)

        self.fc_keys = nn.Linear(self.n_units, n_heads * self.d_k)
        self.fc_values = nn.Linear(self.n_units, n_heads * self.d_k)
        self.fc_query = nn.Linear(self.n_units, n_heads * self.d_k)
        self.W_o = nn.Linear(n_units, n_units)

        self.init_weights()

    def init_weights(self):
        k = np.sqrt(1.0 / self.n_units)

        self.fc_keys.weight = nn.init.uniform_(self.fc_keys.weight, -k, k)
        self.fc_keys.bias = nn.init.uniform_(self.fc_keys.bias, -k, k)

        self.fc_values.weight = nn.init.uniform_(self.fc_values.weight, -k, k)
        self.fc_values.bias = nn.init.uniform_(self.fc_values.bias, -k, k)

        self.fc_query.weight = nn.init.uniform_(self.fc_query.weight, -k, k)
        self.fc_query.bias = nn.init.uniform_(self.fc_query.bias, -k, k)

        self.W_o.weight = nn.init.uniform_(self.W_o.weight, -k, k)
        self.W_o.bias = nn.init.uniform_(self.W_o.bias, -k, k)

        # TODO: create/initialize any necessary parameters or layers
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout

    # NOT used 
    def softmax(self, x):
        val = []
        for idx in range(x.shape[-1]):
            column = x[:,:,:,idx]
            column = column - torch.max(column)

            x_tilde = torch.exp(column)

            summy = torch.sum(column)
            if (torch.sum(torch.isnan(summy)) > 0):
                print("Hello there!")
            val.append(x_tilde/summy)

        
        return torch.stack(val, dim=-1)


    # Based off: https://github.com/harvardnlp/annotated-transformer
    def attention(self, key, query, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        x_tilde = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.float().unsqueeze(1).expand(-1, x_tilde.shape[1], -1, -1) # expands on n_heads
            x_tilde = (x_tilde * mask) - 1e9 * (1-mask)
        
        p_attn = F.softmax(x_tilde, dim = -1)
        #p_attn = self.softmax(x_tilde)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        batch_size = query.size(0)
        seq_len = query.size(1)
        # 1)
        key = self.fc_keys(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        value = self.fc_values(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        query = self.fc_query(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 2)
        x = self.attention(key, query, value, mask=mask, dropout=self.dropout)

        # 3)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_units) # Concat
        output = self.W_o(x)


        return output# size: (batch_size, seq_len, self.n_units)






#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        if (torch.sum(torch.isnan(self.lut(x))).item() > 0):
            print("NAN!!")
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
