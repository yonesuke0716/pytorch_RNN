import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import math
import matplotlib.pyplot as plt
import random
import time
import copy

import pandas as pd  
from pandas_datareader import data as web  
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import RNNModel, LSTMModel, LSTM
from dataset import Mydatasets

#const
epochs = 100
batchs = 32
input_num = 1
hidden_num = 32
output_num = 1
layers_num = 2


def main():
    # dataset
    
    ds_train = Mydatasets(start='2016-5-16', end='2019-12-1')
    ds_val = Mydatasets(start='2019-12-2', end='2021-1-29')

    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=batchs, shuffle=True, num_workers=2, drop_last=True)
    valloader = torch.utils.data.DataLoader(ds_val, batch_size=batchs, shuffle=True, num_workers=2, drop_last=True)

    model = LSTM(n_input=input_num, n_hidden=hidden_num, n_layers=layers_num, n_output=output_num, batch_size=batchs)
    
    loss_train = []
    loss_val = []
    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    best_score = 100
    
    for i in range(epochs):
        #train
        running_loss = 0.0
        iteration = len(trainloader)
        for idx, batch in enumerate(trainloader):
            xs, ys = batch
            xs = torch.transpose(xs, 0, 1)
            # double型→float型
            xs = xs.float()
            ys = ys.float()

            y_pred = model(xs)
            y_pred = torch.transpose(y_pred, 0, 1)

            loss = loss_function(y_pred, ys)
            optimizer.zero_grad()
            running_loss += loss.data
            loss.backward(retain_graph=True)
            optimizer.step()

        loss_train.append(running_loss.item() / iteration)
        
        running_loss = 0.0
        iteration = len(valloader)
        #validation
        for idx, batch in enumerate(valloader):
            xs, ys = batch
            xs = torch.transpose(xs, 0, 1)
            # double型→float型
            xs = xs.float()
            ys = ys.float()
            
            y_pred = model(xs)
            y_pred = torch.transpose(y_pred, 0, 1)

            loss = loss_function(y_pred, ys)
            optimizer.zero_grad()
            running_loss += loss.data
            loss.backward(retain_graph=True)
            optimizer.step()

        loss_val.append(running_loss.item() / iteration)

        if loss_val[i] < best_score:
            best_score = loss_val[i]
            best_model = copy.copy(model)
        torch.save({'state_dict': best_model.state_dict()}, 'best_model.pth')
        
        print('epoch:{} train loss: {}'.format(i, loss_train[i]))
        print('epoch:{} val loss: {}'.format(i, loss_val[i]))
    
    plt.plot(range(epochs), loss_train, label='loss(train)')
    plt.plot(range(epochs), loss_val, label='loss(val)')
    plt.legend()
    plt.grid()
    plt.savefig('loss.jpg')

if __name__ == "__main__":
    # 乱数の固定
    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    print('Start')
    main()
    print('End')
