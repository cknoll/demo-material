"""
This module demonstrates how to model a simple linear time discrete system with pytorch by a recurrent neural network (RNN) and identify the entries of the system matrix only from partial measurement. Because we do not want to have any nonlinearity, we cannot use the rnn-class of pytorch, but instead build up a custom net (see forward method).
"""


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sc
import scipy.linalg

try:
    from ipydex import IPS, activate_ips_on_exception
    activate_ips_on_exception()
except ImportError:
    pass

if 1:
    # achieve reproducible results
    np.random.seed(0)
    torch.manual_seed(0)

class CustomLinearRNN(nn.Module):
    """
    This code is based on https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091 and
    https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79
    """
    def __init__(self, known_state_sz, hidden_state_sz):
        super().__init__()
        self.known_state_sz = known_state_sz
        self.hidden_state_sz = hidden_state_sz

        # weights
        self.A1 = nn.Parameter(torch.Tensor(known_state_sz, known_state_sz))
        self.A2 = nn.Parameter(torch.Tensor(known_state_sz, hidden_state_sz))
        self.A3 = nn.Parameter(torch.Tensor(hidden_state_sz, known_state_sz))
        self.A4 = nn.Parameter(torch.Tensor(hidden_state_sz, hidden_state_sz))
        self.init_weights_random()

    def init_weights_random(self):
        stdv = 1.0 / np.sqrt(self.hidden_state_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def set_weights(self, A):
        self.A1.data = torch.tensor(A[:self.known_state_sz, :self.known_state_sz] + 0.1*0).float()
        self.A2.data = torch.tensor(A[:self.known_state_sz, self.known_state_sz:]).float()
        self.A3.data = torch.tensor(A[self.known_state_sz:, :self.known_state_sz]).float()
        self.A4.data = torch.tensor(A[self.known_state_sz:, self.known_state_sz:]).float()

    def forward(self, x, init_hidden_states):
        """
        Assumes x and init_hidden_states is sorted like the following:
        # sorting: (sequence_index, component_index, batch_index)
        x contains the known states
        """
        seq_sz, feat_sz, batch_sz = x.size()

        hidden_state = init_hidden_states
        known_seq = []
        hidden_seq = []

        for t in range(seq_sz):
            x_t = x[t, :, :] # current known state
            new_known = self.A1@x_t + self.A2@hidden_state
            hidden_state = self.A3@x_t + self.A4@hidden_state

            hidden_seq.append(hidden_state.unsqueeze(0))
            known_seq.append(new_known.unsqueeze(0))


        hidden_seq = torch.cat(hidden_seq, dim=0)
        known_seq = torch.cat(known_seq, dim=0)

        # (maybe= we have to reshape here for bigger systems

        # reshape from shape (sequence, batch, feature) to (sequence, feature, batch)
        #hidden_seq = hidden_seq.transpose(1, 2).contiguous()
        #known_seq = known_seq.transpose(1, 2).contiguous()
        #IPS()

        # apply scaled tanh to prevent inf -> nan for systems that become unstable during training
        k = 1e2

        # return k*torch.tanh(known_seq/k), k*torch.tanh(hidden_seq/k)
        return known_seq, hidden_seq


# define a dynamical syste: here damped harmonic oscillator
A_cont = np.array(
    [[0,     1],
     [-30, -.1]]
     )

# convert from time continuous to time discrete domain
dt = 0.1
A_diskr = sc.linalg.expm(dt*A_cont)





def create_data_batch(xx0, N=30):
    xx = np.zeros((N, 2))
    xx_current = xx0
    for i in range(N):
        xx[i, :] = xx_current
        xx_new = A_diskr@xx_current
        xx_current = xx_new
        # print(xx_current)
    return xx

xx0 = np.array([1, 0])
xx = create_data_batch(xx0)
XX = torch.tensor(xx[:, :, np.newaxis]).float()

# create Training Data:
if 1:
    # create some batches
    tensor_list = []
    for i in range(80):
        xx0 = np.random.random(2)
        xx = create_data_batch(xx0)
        tensor_list.append(torch.tensor(xx[:, :, np.newaxis]).float())
    XX_training = XX = torch.cat(tensor_list, dim=2)



model = CustomLinearRNN(1, 1)


# this is for debugging only: set the weights to the known values
# model.set_weights(A_diskr)


# sorting: (sequence_index, component_index, batch_index)
res = model.forward(x=XX[:, :1, :], init_hidden_states=XX[0, 1:, :])


RR = torch.cat(res, dim=1)

criterion = nn.MSELoss()

loss = criterion(XX[:, 1:, 0], RR[:, :-1, 0])

# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(model.parameters(), lr=0.8)
# training
for i in range(13):
    print('STEP: ', i)
    def closure():
        optimizer.zero_grad()
        res = model.forward(x=XX[:, :1, :], init_hidden_states=XX[0, 1:, :])
        RR = torch.cat(res, dim=1)
        loss = criterion(XX[1:, 0, :], RR[:-1, 0, :])
        print('loss:', loss.item())
        loss.backward()
        return loss
    optimizer.step(closure)



# Create Test_data
if 1:
    # create some batches
    tensor_list = []
    for i in range(80):
        xx0 = np.random.random(2) * 10
        xx = create_data_batch(xx0, N=150)
        tensor_list.append(torch.tensor(xx[:, :, np.newaxis]).float())
    XX_test = XX = torch.cat(tensor_list, dim=2)


# begin to predict, no need to track gradient here
with torch.no_grad():
    res = model.forward(x=XX[:, :1, :], init_hidden_states=XX[0, 1:, :])
    RR = torch.cat(res, dim=1)
    loss = criterion(XX[1:, 0, :], RR[:-1, 0, :])

rr = RR.detach().numpy()
xx = XX.detach().numpy()

# plot all trajectories for all badges
plt.plot(xx[1:, 0, :], color="green", lw=2)
plt.plot(rr[:-1, 0, :], color="blue", linestyle="--")

plt.show()



IPS()
