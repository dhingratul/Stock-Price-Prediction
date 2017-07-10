#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:45:51 2017

@author: dhingratul
"""

import numpy as np
import matplotlib.pyplot as plt


def normalize_windows(win_data):
    """ Normalize a window
    Input: Window Data
    Output: Normalized Window

    Note: Run from load_data()

    Note: Normalization data using n_i = (p_i / p_0) - 1,
    denormalization using p_i = p_0(n_i + 1)
    """
    norm_data = []
    for w in win_data:
        norm_win = [((float(p) / float(w[0])) - 1) for p in w]
        norm_data.append(norm_win)
    return norm_data


def load_data(filename, seq_len, norm_win):
    """
    Loads the data from a csv file into arrays

    Input: Filename, sequence Lenght, normalization window(True, False)
    Output: X_tr, Y_tr, X_te, Y_te

    Note: Normalization data using n_i = (p_i / p_0) - 1,
    denormalization using p_i = p_0(n_i + 1)

    Note: Run from timeSeriesPredict.py
    """
    fid = open(filename, 'r').read()
    data = fid.split('\n')
    sequence_length = seq_len + 1
    out = []
    for i in range(len(data) - sequence_length):
        out.append(data[i: i + sequence_length])
    if norm_win:
        out = normalize_windows(out)
    out = np.array(out)
    split_ratio = 0.9
    split = round(split_ratio * out.shape[0])
    train = out[:int(split), :]
    np.random.shuffle(train)
    X_tr = train[:, :-1]
    Y_tr = train[:, -1]
    X_te = out[int(split):, :-1]
    Y_te = out[int(split):, -1]
    X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], 1))
    X_te = np.reshape(X_te, (X_te.shape[0], X_te.shape[1], 1))
    return [X_tr, Y_tr, X_te, Y_te]


def predict_seq_mul(model, data, win_size, pred_len):
    """
    Predicts multiple sequences
    Input: keras model, testing data, window size, prediction length
    Output: Predicted sequence

    Note: Run from timeSeriesPredict.py
    """
    pred_seq = []
    for i in range(len(data)//pred_len):
        current = data[i * pred_len]
        predicted = []
        for j in range(pred_len):
            predicted.append(model.predict(current[None, :, :])[0, 0])
            current = current[1:]
            current = np.insert(current, [win_size - 1], predicted[-1], axis=0)
        pred_seq.append(predicted)
    return pred_seq


def plot_mul(Y_hat, Y, pred_len):
    """
    PLots the predicted data versus true data

    Input: Predicted data, True Data, Length of prediction
    Output: return plot

    Note: Run from timeSeriesPredict.py
    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(Y, label='Y')
    # Print the predictions in its respective series-length
    for i, j in enumerate(Y_hat):
        shift = [None for p in range(i * pred_len)]
        plt.plot(shift + j, label='Y_hat')
        plt.legend()
    plt.show()
