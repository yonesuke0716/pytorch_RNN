import numpy as np
import math

def mkDataSet(data_size, data_length=50, freq=60., noise=0.02):
    """
    params
      data_size : データセットサイズ
      data_length : 各データの時系列長
      freq : 周波数
      noise : ノイズの振幅
    returns
      train_x : トレーニングデータ（t=1,2,...,size-1の値)
      train_t : トレーニングデータのラベル（t=sizeの値）
    """
    train_x = []
    train_t = []

    for offset in range(data_size):
        train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])

    return train_x, train_t

def sin_dataset(n_data, n_test):
    x = np.linspace(0, 2 * np.pi, n_data + n_test)
    ram = np.random.permutation(n_data + n_test)
    x_train = np.sort(x[ram[:n_data]])
    x_test = np.sort(x[ram[n_data:]])

    y_train = np.sin(x_train)
    y_test = np.sin(x_test)

    return x_train, y_train, x_test, y_test