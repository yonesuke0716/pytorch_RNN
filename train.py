import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import matplotlib.pyplot as plt
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import RNNModel, LSTMModel
from dataset import sin_dataset

# parameter
epochs = 300
n_input = 1
n_hidden = 50
n_output = 1
num_layers = 1
n_batch = 20
n_data = 1000
n_test = 200


def main():
    # dataset
    x_train, y_train, x_test, y_test = sin_dataset(n_data, n_test)
    y_test_torch = torch.from_numpy(np.asarray(y_test))
    y_test_torch = y_test_torch.unsqueeze(0)
    """
    training_size = 10000
    test_size = 1000
    epochs_num = 1000
    hidden_size = 5
    batch_size = 100
    train_x, train_t = mkDataSet(training_size)
    test_x, test_t = mkDataSet(test_size)
    """

    # model, optimizer, loss function
    model = LSTMModel(n_input, n_hidden, n_output, num_layers)
    optimizer = optim.Adam(model.parameters())
    MSE = nn.MSELoss()
    # MAE = nn.L1Loss()

    train_loss = []
    test_loss = []

    if os.path.exists("result/loss") == False:
        os.makedirs("result/loss")
    if os.path.exists("result/eval") == False:
        os.makedirs("result/eval")
    if os.path.exists("result/model") == False:
        os.makedirs("result/model")

    # train
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_data)
        sum_loss = 0
        for i in range(0, n_data, n_batch):
            # batch化
            data = x_train[perm[i : i + n_batch]]
            label = y_train[perm[i : i + n_batch]]
            label = torch.from_numpy(label).double()
            label = label.unsqueeze(0)

            # optimizerの初期化
            optimizer.zero_grad()
            output = model(data, label).double()

            loss = MSE(output, label)
            loss.backward()
            optimizer.step()
            sum_loss += loss.data * n_batch

        # loss(train)
        ave_loss = sum_loss / n_data
        train_loss.append(ave_loss)

        # loss(val)
        model.eval()
        y_test_pred = model(x_test)
        loss = MSE(y_test_pred, y_test_torch)
        test_loss.append(loss.data)

        # loss display
        if epoch % 100 == 1:
            print("Ep/MaxEp     train_loss     test_loss")

        if epoch % 10 == 0:
            print(
                "{:4}/{}  {:10.5}   {:10.5}".format(
                    epoch, epochs, ave_loss, float(loss.data)
                )
            )

        if epoch % 20 == 0:
            # epoch20ごとのテスト予測結果
            plt.figure(figsize=(5, 4))

            # 時間の計測
            start = time.time()
            y_pred = model.forward(x_test)
            end = time.time() - start
            print("処理時間：{}sec".format(end))

            # tensor → numpy
            y_pred = y_pred.to("cpu").detach().numpy().copy()
            plt.plot(x_test, y_test, label="target")
            plt.plot(x_test, y_pred[0], label="predict")
            plt.legend()
            plt.grid(True)
            plt.xlim(0, 2 * np.pi)
            plt.ylim(-1.2, 1.2)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig("result/eval/ep{}.png".format(epoch))
            plt.clf()
            plt.close()

    # save loss glaph
    plt.figure(figsize=(5, 4))
    plt.plot(train_loss, label="training")
    plt.plot(test_loss, label="test")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.savefig("result/loss/loss_history.png")
    plt.clf()
    plt.close()

    # save best_model
    torch.save(model, "result/model/best_model.pt")


if __name__ == "__main__":
    # 乱数の固定
    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    print("Start")
    main()
    print("End")
