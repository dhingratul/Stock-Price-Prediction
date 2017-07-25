#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:54:11 2017

@author: dhingratul

Predicts the next day (closing) stock prices for S&P 500 data using LSTM,
and 1D conv layer
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import helper
import time
from sklearn.metrics import mean_squared_error
import numpy as np

# Load Data
seq_len = 50
norm_win = True
filename = '../data/sp500.csv'
X_tr, Y_tr, X_te, Y_te = helper.load_data(filename, seq_len, norm_win)
# Model Build
model = Sequential()
model.add(LSTM(input_dim=1,
               output_dim=seq_len,
               return_sequences=True))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100,
               return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_dim=1))  # Linear dense layer to aggregate into 1 val
model.add(Activation('linear'))
timer_start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('Model built in: ', time.time()-timer_start)
# Training model
model.fit(X_tr,
          Y_tr,
          batch_size=512,
          nb_epoch=100,
          validation_split=0.05
          )
# Predictions
win_size = seq_len
pred_len = seq_len
plot = False
if plot:
    pred = helper.predict_seq_mul(model, X_te, win_size, pred_len)
    helper.plot_mul(pred, Y_te, pred_len)
else:
    pred = helper.predict_pt_pt(model, X_te)
    mse_model = mean_squared_error(Y_te, pred)
    print("MSE of model ", mse_model)
    # Stupid Model
    y_bar = np.mean(X_te, axis=1)
    y_bar = np.reshape(y_bar, (y_bar.shape[0]))
    mse_base = mean_squared_error(Y_te, y_bar)
    print("MSE of Base Model", mse_base)
    # t-1 Model
    y_t_1 = X_te[:, -1]
    y_t_1 = np.reshape(y_t_1, (y_t_1.shape[0]))
    mse_t_1 = mean_squared_error(Y_te, y_t_1)
    print("MSE of t-1 Model", mse_t_1)
    # Comparisons
    improv = (mse_model - mse_base)/mse_base
    improv_t_1 = (mse_model - mse_t_1)/mse_base
    print("%ge improvement over naive model", improv)
    print("%ge improvement over t-1 model", improv_t_1)
    corr_model = np.corrcoef(Y_te, pred)
    corr_base = np.corrcoef(Y_te, y_bar)
    corr_t_1 = np.corrcoef(Y_te, y_t_1)
    print("Correlation of base, vs naive model vs t-1 model \n", corr_base,
          "\n \n", corr_model, "\n \n", corr_t_1)
