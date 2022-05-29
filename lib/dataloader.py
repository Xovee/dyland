import os

import numpy as np
import torch
from torch.utils.data import Dataset

import lib.toy_data as toy_data
from lib.toy_data import generate_slope


def load_features_labels(file="use_info.csv", train_rate=0.5, val_rate=0.3, seq_len=3, pre_len=1, all=False):
    if "YC" in os.path.basename(os.path.splitext(file)[0]):
        data = np.loadtxt(fname=file, skiprows=1, delimiter=",")[:, 3:]
    else:
        data = np.loadtxt(fname=file, skiprows=0)[:, 3:]
    data = data.transpose()  # shape=(time_sequence, number_of_nodes)
    print("Shape of {}: {} ".format(file, data.shape))
    num_nodes = data.shape[1]
    max_value = np.max(data)
    data_normal = data / max_value

    train_size = int(data_normal.shape[0] * train_rate)
    val_size = int(data_normal.shape[0] * val_rate)
    train_data = data_normal[0: train_size]
    val_data = data_normal[train_size: train_size + val_size]
    test_data = data_normal[train_size + val_size:]

    def get_X_and_Y(data):
        X, Y = [], []
        for i in range(len(data) - seq_len - pre_len + 1):
            X.append(data[i: i + seq_len])
            Y.append(data[i + seq_len: i + seq_len + pre_len])
        return np.array(X), np.array(Y)

    trainX, trainY = get_X_and_Y(train_data)
    valX, valY = get_X_and_Y(val_data)
    testX, testY = get_X_and_Y(test_data)
    if all:
        all_X, all_Y = get_X_and_Y(data_normal)

    print("Shape of trainX: {} trainY: {}".format(trainX.shape, trainY.shape))
    print("Shape of valX: {} valY: {}".format(valX.shape, valY.shape))
    print("Shape of testX: {} testY: {}".format(testX.shape, testY.shape))

    if all:
        return num_nodes, max_value, trainX, trainY, valX, valY, testX, testY, all_X, all_Y
    else:
        return num_nodes, max_value, trainX, trainY, valX, valY, testX, testY


def load_loc_data(file_name=None, first_three=True, batch_size=100, data_path="data"):
    """
    toy_slope : generate new slope every call.
    """
    if file_name in ['2spirals_1d', '2spirals_2d', 'swissroll_1d', 'swissroll_2d', 'circles_1d', 'circles_2d', '2sines_1d',
                 'target_1d']:
        x = toy_data.inf_train_gen(file_name, batch_size=batch_size)
        return x
    elif file_name in ["random_slope", "random_slope_flat"]:
        train_data = generate_slope(n=batch_size)
    elif "YC" in file_name or "DDH" in file_name:
        file_name = os.path.join(data_path, file_name + ".csv")
        train_data = np.loadtxt(fname=file_name, skiprows=1, delimiter=",")[:, 0:3]
        # max_min normalization
        for i in [0, 1, 2]:
            h = train_data[:, i]
            temp = (h - h.min()) / (h.max() - h.min())
            train_data[:, i] = temp
    return train_data


class MyDataSet(Dataset):
    def __init__(self, data_path, seq_len=3, pre_len=1, device='cuda:0'):
        self.seq_len = seq_len
        self.pre_len = pre_len
        data = np.loadtxt(fname=data_path, skiprows=1, delimiter=",")[:, 3:]
        # shape=(time_sequence, number_of_nodes)
        data = data.transpose()
        print("Shape of {}: {} ".format(data_path, data.shape))
        self.num_nodes = data.shape[1]
        data_normal = self.max_min_norm(data)
        self.trainX, self.trainY = self.get_X_and_Y(data_normal)
        self.trainX = torch.from_numpy(self.trainX.transpose(0, 2, 1)).float().to(device)
        self.trainY = torch.from_numpy(self.trainY.transpose(0, 2, 1)).float().to(device)
        self.len = self.trainX.shape[0]

    def __getitem__(self, index):
        x = self.trainX[index]
        y = self.trainY[index]
        return x, y

    def __len__(self):
        return self.len

    @staticmethod
    def max_min_norm(data):
        max_value = np.max(data)
        min_value = np.min(data)
        return (data - min_value)/(max_value - min_value)

    def get_X_and_Y(self, data):
        X, Y = [], []
        for i in range(len(data) - self.seq_len - self.pre_len + 1):
            X.append(data[i: i + self.seq_len])
            Y.append(data[i + self.seq_len: i + self.seq_len + self.pre_len])
        return np.array(X), np.array(Y)

