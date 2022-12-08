'''
Script used to load data, compile NN, and train NN for guessing a player's ELO
based on chess games they have played. 

Written by Robert Bennett for CS 230.

An early version of this script was based off of the following YouTube tutorial:
https://www.youtube.com/watch?v=c0k-YLQGKjY

The present script should be extremely different from the above tutorial, 
with perhaps some vestigal bits remaining, e.g. in the import statements. I'm
just disclosing the tutorial I used to make it clear I'm not trying to secretly
copy it.
'''

###############################################################################
#
# Import and define functions
#
###############################################################################

import chess_NN_helper_funcs as cf
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd                                                             
import random
import scipy.stats
from sklearn.utils import shuffle
import tensorflow as tf                                                         
from tensorflow.keras.models import Sequential                                  
from tensorflow.keras.layers  import *                                          
from tensorflow.keras.callbacks import ModelCheckpoint                          
from tensorflow.keras.losses import MeanSquaredError                            
from tensorflow.keras.metrics import RootMeanSquaredError                       
from tensorflow.keras.optimizers import Adam                                    
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import time

###############################################################################
#
# Initialize
#
###############################################################################

dir_path = os.path.dirname(os.path.abspath(__file__)) 
data_file_base  = '2022-07_600+0_{}_evals_processed.dat'
data_file_base = '/home/rob/Documents/sync/PhD/Courses/CS_230/'+\
                  'project/data_07/' + data_file_base

raw_data = []
X_initial = []
Y_initial = []
min_length = 40
parts =  ['00']#, '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']


###############################################################################
#
# Load data
#
###############################################################################

file_list = []
for part in parts:
    file_list.append(data_file_base.format(part))
X_data, Y_data = cf.load_data(file_list)
random.seed(2**123 - 1)
Z = list(zip(X_data,Y_data))
random.shuffle(Z)
X_data, Y_data = zip(*Z)

print('Main data loaded!')

###############################################################################
#
# Process into training, test, and dev sets
#
###############################################################################

train_size = 0.95
test_size = 0.025
dev_size = 0.025
num_vals = len(X_data)

train_size_end = int(num_vals*train_size)
test_size_end = int(num_vals*(train_size + test_size))

#X = np.asarray(X).astype('float32')
X_train, Y_train = X_data[0:train_size_end], Y_data[0:train_size_end]
X_test, Y_test = X_data[train_size_end:test_size_end],\
                 Y_data[train_size_end:test_size_end]
X_dev, Y_dev = X_data[test_size_end::], Y_data[test_size_end::]

###############################################################################
#
# Load additional data (containing high and low elos) and add to training set
#
###############################################################################

counter = 0
additional_parts = [
                    '00'#, '01', '02', '03', '04', '05', 
                    #'06', '07', '08', '09', '10'
                    ]
additional_data_file_list = []
data_file_base  = '2022-08_600+0_{}_evals_processed_lessthan1300_'+\
                  'greaterthan1900.dat'
data_file_base = '/home/rob/Documents/sync/PhD/Courses/CS_230/'\
                 + 'project/data_08_additional/' + data_file_base

for part in additional_parts:
    additional_data_file_list.append(data_file_base.format(part))
X_additional, Y_additional = cf.load_data(additional_data_file_list)

random.seed(2**123 - 1)

X_train = list(X_train) + X_additional
Y_train = list(Y_train) + Y_additional

Z_train = list(zip(X_train,Y_train))
random.shuffle(Z_train)
X_train, Y_train = zip(*Z_train)

X_additional = []
Y_additional = []

print('Additional data loaded!')

###############################################################################

# Process all inputs into normalized arrays

###############################################################################

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)
X_dev, Y_dev = np.array(X_dev), np.array(Y_dev)

X_train = X_train.reshape(X_train.shape[0], min_length, X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], min_length, X_test.shape[2])
X_dev = X_dev.reshape(X_dev.shape[0], min_length, X_dev.shape[2])
X_scaling = []

for i in range(X_train.shape[2]):
    mean_val = np.mean(X_train[:,:,i])
    stdev_val = np.std(X_train[:,:,i])
    X_train[:,:,i] = (X_train[:,:,i] - mean_val) / stdev_val
    X_test[:,:,i] = (X_test[:,:,i] - mean_val) / stdev_val
    X_dev[:,:,i] = (X_dev[:,:,i] - mean_val) / stdev_val
    X_scaling.append((mean_val, stdev_val))

Ymean = np.mean(Y_train)
Ystdev = np.std(Y_train)

Y_train = (Y_train - Ymean)/(Ystdev)
Y_test = (Y_test - Ymean)/(Ystdev)
Y_dev = (Y_dev - Ymean)/(Ystdev)

Ymax1 = np.max(Y_train)
Ymin1 = np.min(Y_train)

Ymax2 = np.max(Y_test)
Ymin2 = np.min(Y_test)

Ymax3 = np.max(Y_dev)
Ymin3 = np.min(Y_dev)

Ymax = np.max([Ymax1, Ymax2, Ymax3])
Ymin = np.min([Ymin1, Ymin2, Ymin3])

Ycenter = (Ymax + Ymin)/2

Y_train = (Y_train-Ycenter)/Ymax
Y_test = (Y_test-Ycenter)/Ymax
Y_dev = (Y_dev-Ycenter)/Ymax

print(np.max(Y_train))
print(np.min(Y_train))

np.savetxt(dir_path + '/X_scaling_info.dat', np.array(X_scaling))
np.savetxt(dir_path + '/Y_scaling_info.dat', np.array([Ymean, Ystdev, Ymax, 
                                                       Ymin, Ycenter]))

print(np.shape(X_test), np.size(Y_test))

###############################################################################
#
# Build Neural Net
#
###############################################################################

# builds the model from the helper function (preloads a trained model if 
# True, otherwise builds our best-performing NN from scratcuh)
load_model_bool = False
modelname = 'Model'
model = cf.build_model(load_model_bool, X_train.shape[2], modelname = modelname) 
model.summary()

learning_rate = 8.8e-4
batch_size = 256

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
cp = ModelCheckpoint(modelname, save_best_only=True)

model.compile(
              loss=MeanSquaredError(), 
              optimizer=Adam(learning_rate=learning_rate), 
              metrics=[RootMeanSquaredError()]
              )

###############################################################################
#
# Train and test model
#
###############################################################################

model_fit = model.fit(
                      X_train, 
                      Y_train, 
                      validation_data=(X_dev,Y_dev), 
                      epochs=100,
                      callbacks=[cp, early_stopping], 
                      batch_size = batch_size
                      )

# Extract relevant quantities from model fit
fit_data = model_fit.history
RMSE = fit_data['root_mean_squared_error']
val_RMSE = fit_data['val_root_mean_squared_error']
test_predictions = np.array(model.predict(X_test).flatten())

###############################################################################
#
# Plot and write relevant information to file
#
###############################################################################

plt.plot(RMSE, color = 'k', ls = '-')
plt.plot(val_RMSE, color = 'r', ls = '--')
plt.ylim(0,1)
plt.savefig('RMS_error_all.pdf'.format(i))
plt.close()

plt.plot(RMSE, color = 'k', ls = '-')
plt.plot(val_RMSE, color = 'r', ls = '--')
plt.ylim(0,0.18)
plt.savefig('RMS_error_zoomed.pdf'.format(i))
plt.close()

# write relevant information to file. Here, we also write details of the model
# just as a sanity check so we can make sure that we ran our model with the
# correct settings
with open ('results.dat', 'a') as f:
    f.write('Train set RMSE: {}'.format(RMSE))
    f.write('\n')
    f.write('Dev set RMSE: {}'.format(val_RMSE))
    f.write('\n')
