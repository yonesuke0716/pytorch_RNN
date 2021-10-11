import numpy as np
import math
import pandas as pd  
from pandas_datareader import data as web
import pandas_datareader as pdr

from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, transform=None, start='2016-5-16', end='2021-1-8', data_len=5):
        self.transform = transform
        
        df_stock = web.DataReader('PG', data_source='yahoo', start=start, end=end)
        df_GS = pdr.get_data_fred('DGS10')
        
        df_concat = pd.concat([df_stock, df_GS], axis=1)
        df = df_concat.dropna()

        # 正規化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df)
        np_norm = scaler.transform(df)
        idx = df.index
        col = df.columns
        df_norm = pd.DataFrame(data=np_norm, index=idx, columns=col)

        #self.data = df_norm.drop(['Adj Close'], axis=1)
        self.data = df_norm['DGS10']
        self.label = df_norm['Adj Close']
        
        # self.data = self.data.fillna(0)
        # self.label = self.label.fillna(0)
        

        self.datanum = len(df.index)

        self.iter_num = int(self.datanum / data_len)

        # データシーケンス作る
        self.xs = []
        self.ys = []
        for i in range(self.iter_num):
            #x = self.data.iloc[i*data_len:(i+1)*data_len, :]
            x = self.data.iloc[i*data_len:(i+1)*data_len]
            # 要素がひとつの場合はSeriesなので、DataFrameへ変更
            x = pd.DataFrame(x)
            x = torch.tensor(x.values)
            y = self.label.iloc[i*data_len:(i+1)*data_len]
            # 要素がひとつの場合はSeriesなので、DataFrameへ変更
            y = pd.DataFrame(y)
            y = torch.tensor(y.values)
            self.xs.append(x)
            self.ys.append(y)

    def __len__(self):
        return self.iter_num

    def __getitem__(self, idx):
        out_data = self.xs[idx]
        out_label = self.ys[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

    def get_dataset(self):
        return self.data, self.label
