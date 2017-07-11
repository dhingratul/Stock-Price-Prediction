#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:54:11 2017

@author: dhingratul
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import helper
import time
from sklearn.metrics import mean_squared_error

# Load Data
seq_len = 25
norm_win = True
filename = 'sp500.csv'
X_tr, Y_tr, X_te, Y_te = helper.load_data(filename, seq_len, norm_win)
# Model Build
model = Sequential()
model.add(LSTM(input_dim=1,
               output_dim=seq_len,
               return_sequences=True))
model.add(Dropout(0.2))
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
          nb_epoch=1,
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
    print("MSE is ", mean_squared_error(Y_te, pred))
