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
layers_num = 1


def main():
    # dataset
    ds_test = Mydatasets(start='2020-1-30', end='2021-2-26')

    testloader = torch.utils.data.DataLoader(ds_test, batch_size=batchs, shuffle=False, num_workers=2, drop_last=True)

    model = LSTM(n_input=input_num, n_hidden=hidden_num, n_layers=layers_num, n_output=output_num, batch_size=batchs)
    model.load_state_dict(torch.load('best_model.pth'), strict=False)

    pred_list = []
    label_list = []
    for idx, batch in enumerate(testloader):
        xs, ys = batch
        xs = torch.transpose(xs, 0, 1)
        # double型→float型
        xs = xs.float()
        ys = ys.float()
            
        y_pred = model(xs)
        y_pred = torch.transpose(y_pred, 0, 1)

        ys = ys.squeeze().detach().numpy()
        y_pred = y_pred.squeeze().detach().numpy()
        for i in range(len(ys)):
            for predict in y_pred[i]:
                pred_list.append(predict)
            for label in ys[i]:
                label_list.append(label)
    
    """
    len_y = ys.size
    y_test = y_test[-len_y:]
    ys = ys[-len_y:]
    """
    
    df = pd.DataFrame({'predict':pred_list,'target':label_list})

    df.to_csv("result.csv", index=False)
    

if __name__ == "__main__":
    # 乱数の固定
    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    print('Start')
    main()
    print('End')
