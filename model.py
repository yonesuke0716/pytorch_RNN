import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNNHardCell(nn.Module):
    def __init__(self, n_input:int, n_hidden:int, state=None) -> None:
        super(RNNHardCell, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.in_h = nn.Linear(self.n_input, self.n_hidden, bias=False)
        self.h_h = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.state = state
        self.register_parameter()

    def register_parameter(self) -> None:
        stdv = 1.0 / math.sqrt(self.n_hidden)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, x, state=None):
        self.state = state
        if self.state is None:
            #self.state = torch.tanh(self.in_h(x))
            #self.state = F.hardtanh(self.in_h(x))
            self.state = F.relu(self.in_h(x))

        else:
            #self.state = torch.tanh(self.in_h(x) + self.h_h(self.state))
            #self.state = F.hardtanh(self.in_h(x) + self.h_h(self.state))
            self.state = F.relu(self.in_h(x) + self.h_h(self.state))
        return self.state

class RNNModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = RNNHardCell(n_input, n_hidden)
        #self.rnn = nn.RNN(n_input, n_hidden, num_layers, nonlinearity='tanh')
        self.out = nn.Linear(n_hidden, n_output, bias=False)
        self.num_layers = num_layers
        
    def forward(self, xs, state=None):
        state = None
        h_seq = []
        for x in xs:
            x = torch.from_numpy(np.asarray(x)).float()
            x = x.unsqueeze(0)
            for _ in range(self.num_layers):
                state = self.rnn(x, state)
            h_seq.append(state)
        
        h_seq = torch.stack(h_seq)
        ys = self.out(h_seq)
        ys = torch.transpose(ys, 0, 1)

        return ys


class LSTMHardCell(nn.Module):
    def __init__(self, n_input:int, n_hidden:int, state=None, cell=None) -> None:
        super(LSTMHardCell, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.in_f = nn.Linear(self.n_input, self.n_hidden, bias=False)
        self.in_i = nn.Linear(self.n_input, self.n_hidden, bias=False)
        self.in_o = nn.Linear(self.n_input, self.n_hidden, bias=False)
        self.in_u = nn.Linear(self.n_input, self.n_hidden, bias=False)
        self.h_f = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.h_i = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.h_o = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.h_u = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.state = state
        self.cell = cell
        self.register_parameter()

    def register_parameter(self) -> None:
        stdv = 1.0 / math.sqrt(self.n_hidden)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, x, state=None, cell=None):
        self.state = state
        self.cell = cell
        if self.state is None:
            f = torch.sigmoid(self.in_f(x))
            i = torch.sigmoid(self.in_i(x))
            o = torch.sigmoid(self.in_o(x))
            u = F.hardtanh(self.in_u(x))
            #u = torch.tanh(self.in_u(x))
        else:
            f = torch.sigmoid(self.in_f(x) + self.h_f(self.state))
            i = torch.sigmoid(self.in_i(x) + self.h_i(self.state))
            o = torch.sigmoid(self.in_o(x) + self.h_o(self.state))
            u = F.hardtanh(self.in_u(x) + self.h_u(self.state))
            #u = torch.tanh(self.in_u(x) + self.h_u(self.state))
        if self.cell is None:
            self.cell = (i * u)
        else:
            self.cell = (f * self.cell) + (i * u)
            
        self.state = o * F.hardtanh(self.cell)
        #self.state = o * torch.tanh(self.cell)

        return self.state, self.cell


class LSTMModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, num_layers=1):
        super(LSTMModel, self).__init__()
        self.rnn = LSTMHardCell(n_input, n_hidden)
        self.out = nn.Linear(n_hidden, n_output, bias=False)
        self.num_layers = num_layers
        
    def forward(self, xs, state=None, cell=None):
        state = None
        cell = None
        h_seq = []
        
        for x in xs:
            x = torch.from_numpy(np.asarray(x)).float()
            x = x.unsqueeze(0)
            for _ in range(self.num_layers):
                state, cell = self.rnn(x, state, cell)
            h_seq.append(state)
        
        h_seq = torch.stack(h_seq)
        ys = self.out(h_seq)
        ys = torch.transpose(ys, 0, 1)

        return ys