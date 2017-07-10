#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:45:51 2017

@author: dhingratul
"""

import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def normalize_windows(win):
    """ Normalize a window
    Input: Window Data
    Output: Normalized Window

    Note: Run from load_data()

    Note: Normalization data using n_i = (p_i / p_0) - 1,
    denormalization using p_i = p_0(n_i + 1)
    """
    norm_data = []
    for w in win:
        norm_win = [((float(p) / float(w[0])) - 1) for p in w]
        norm_data.append(norm_win)
        return norm_data


def load_data(fname, seq_len, norm_win):
    """
    Loads the data from a csv file into arrays

    Input: Filename, sequence Lenght, normalization window(True, False)
    Output: X_tr, Y_tr, X_te, Y_te

    Note: Normalization data using n_i = (p_i / p_0) - 1,
    denormalization using p_i = p_0(n_i + 1)
    """
    fid = open(fname, 'r').read()
    data = fid.split('\n')
    sequence_length = seq_len + 1
    out = []
    for i in range(len(data) - sequence_length):
        out.append(data[i: i + sequence_length])
    if norm_win:
        out = normalize_windows(out)
    out = np.array(out)
    split = round(0.9 * out.shape[0])
    train = out[:int(split), :]
    np.random.shuffle(train)
    X_tr = train[:, :-1]
    Y_tr = train[:, -1]
    X_te = out[int(split):, :-1]
    Y_te = out[int(split):, -1]
    X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], 1))
    X_te = np.reshape(X_te, (X_te.shape[0], X_te.shape[1], 1))
    return [X_tr, Y_tr, X_te, Y_te]
