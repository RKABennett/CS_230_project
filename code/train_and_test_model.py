###############################################################################
#
# Import and define functions
#
###############################################################################

import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd                                                             
import tensorflow as tf                                                         
from tensorflow.keras.models import Sequential                                  
from tensorflow.keras.layers  import *                                          
from tensorflow.keras.callbacks import ModelCheckpoint                          
from tensorflow.keras.losses import MeanSquaredError                            
from tensorflow.keras.metrics import RootMeanSquaredError                       
from tensorflow.keras.optimizers import Adam                                    
from tensorflow.keras.models import load_model
import time
import random
import scipy.stats
from sklearn.utils import shuffle


def calc_error(prediction, actual):
    '''
    Calculates the RMS and mean square error of a prediction compared to the
    actual dataset.
    '''
    error_array = prediction - actual
    mean_sq_error = np.sum(error_array**2)/(np.size(error_array))
    RMS_error = np.sqrt(np.mean(error_array**2))
    return RMS_error, mean_sq_error

def load_data(file_list):
    '''
    Load into memory every game matrix X that are saved in a series of files,
    given by file_list. 

    Returns:

    X: A 3D array containing all of our game matrices.
    Y: A 1D array containing the average elo of the players for each 
    game matrix in X. 
    '''

    X = []
    Y = []

    for data_file in file_list:
        with open(data_file) as file:
            counter = 0 # for keeping track of which line we are in
            for line in file:
                counter += 1
                game_matrix = []
                
                if counter == 1:
                    # header of our game file
                    game_win = 0
                    game_normal = 0
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    line = line.replace('\n', '')
                    line = line.replace('\'', '')
                    line = line.split(',')
                    
                    # find the winner of the game and how it ended
                    for attribute in line:
                        if 'Termination' in attribute:
                            termination = attribute
                        if 'Result' in attribute:
                            result = attribute
                    if 'Normal' in termination:
                        game_normal = 1
                    if '1-0' in result:
                        game_win = 1
                    elif '0-1' in result:
                        game_win = -1
                    else:
                        game_win = 0
                    
                # find the elo of the players in the game
                if counter == 2:
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    line = line.replace('\n', '')
                    line = line.replace('\'', '')
                    line = line.split(',')
                    elo = np.mean([float(i) for i in line])
                    elo_diff = float(line[0]) - float(line[1])
                elif counter == 3:
                    # find the engine eval for each move
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    line = line.replace('\n', '')
                    line = line.split(',')
                    eval_list = ([float(i) for i in line]) 

                elif counter == 4:
                    # find the clock time for each move
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    line = line.replace('\n', '')
                    line = line.split(',')
                    time_list = ([float(i) for i in line]) 

                elif counter == 5:
                    counter = 0 # reset for the next game
                    line = ast.literal_eval(line)
                    for i, move in enumerate(line):
                        # move through the list of moves to build our move
                        # vector M for each move.
                        iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo, 
                        isking, isqueen, isrook, isbishop, isknight, ispawn, 
                        row, col = move2vec(move)
                        

                        # compile the move vector and append it to the game 
                        # matrix.

                        try:
                            cpl = eval_list[i]
                        except:
                            cpl = -1
                        time_spent = time_list[i]
                        move_vector = np.array([cpl, time_spent, game_win, game_normal, iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo, isking, isqueen, isrook, isbishop, isknight, ispawn, row, col])
                        game_matrix.append(move_vector.T)
                    
                    # add the game matrix to X if it contains at least min_length moves.
                    if len(game_matrix) >= min_length:
                        X.append(game_matrix[:min_length])
                        Y.append(elo)

    return X,Y


def move2vec(move):
    
    # see if our move already exists in our move dictionary. If it does, 
    # return it right away and exit the function.
    try:
        iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo, isking, isqueen, 
        isrook, isbishop, isknight, ispawn, row, col = move_dictionary[move]
        return iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo, 
        isking, isqueen, isrook, isbishop, isknight, ispawn, row, col
    except:
        pass

    # if the move is not in the move dictionary, we build an entry for it by
    # manually going through the move and picking out the salient features.
    # see a description of the move vector M in the report for an explanation
    # of what each term here means.
    isking = 0
    isqueen = 0
    isrook = 0
    isbishop = 0
    isknight = 0
    ispawn = 0
    
    ispromo = 0
    ischeck = 0
    iscastle_ks = 0
    iscastle_qs = 0
    iscapture = 0
    
    filtered_move = move.replace('+', '').replace('x', '').replace('#', '').replace('=','')
    try:
        float(filtered_move[-1])
    except:
        filtered_move = filtered_move[0:-1]

    if not 'O' in filtered_move:
        row, col = square_dictionary[filtered_move[-2] + filtered_move[-1]]
    else:
        row, col = -1, -1

    if '=' in move:
        ispromo = 1
    elif 'O-O-O' in move:
        iscastle_qs = 1
        isking = 1
        isrook = 1
    elif 'O-O' in move:
        iscastle_ks = 1
        isking = 1
        isrook = 1
    elif 'K' in move:
        isking = 1
    elif 'Q' in move:
        isqueen = 1
    elif 'R' in move:
        isrook = 1
    elif 'B' in move:
        isbishop = 1
    elif 'N' in move:
        isknight = 1
    else:
        ispawn = 1

    if '+' in move:
        ischeck = 1
    if 'x' in move:
        iscapture = 1

    # add the move to our move dictionary and return it. 
    move_dictionary[move] = iscapture, ischeck, iscastle_ks, iscastle_qs, 
    ispromo, isking, isqueen, isrook, isbishop, isknight, ispawn, row, col
    return iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo, isking, 
    isqueen, isrook, isbishop, isknight, ispawn, row, col


###############################################################################
#
# Initialize
#
###############################################################################

dir_path = os.path.dirname(os.path.abspath(__file__)) 
data_file_base  = '2022-07_600+0_{}_evals_processed.dat'
data_file_base = dir_path + '/../data_07/' + data_file_base

raw_data = []
X_initial = []
Y_initial = []
min_length = 38
parts =  ['00']#, '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

move_dictionary = dict()
square_dictionary = dict()
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8']

for i in range(8):
    for j in range(0,8):
        square_l = letters[i]
        square_n = numbers[j]
        square_dictionary[square_l + square_n] = (i,j) 

###############################################################################
#
# Load data
#
###############################################################################

file_list = []
for part in parts:
    file_list.append(data_file_base.format(part))
X_data, Y_data = load_data(file_list)
random.seed(2**1234 - 1)
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
X_test, Y_test = X_data[train_size_end:test_size_end], Y_data[train_size_end:test_size_end]
X_dev, Y_dev = X_data[test_size_end::], Y_data[test_size_end::]

###############################################################################
#
# Load additional data (containing high and low elos) and add to training set
#
###############################################################################
counter = 0
additional_parts = ['00']#, '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
additional_data_file_list = []
data_file_base  = '2022-08_600+0_{}_evals_processed_lessthan1300_greaterthan1900.dat'
data_file_base = dir_path + '/../data_08_additional/' + data_file_base

for part in additional_parts:
    additional_data_file_list.append(data_file_base.format(part))
X_additional, Y_additional = load_data(additional_data_file_list)

random.seed(2**1234 - 1)

X_train = list(X_train) + X_additional
Y_train = list(Y_train) + Y_additional

Z_train = list(zip(X_train,Y_train))
random.shuffle(Z_train)
X_train, Y_train = zip(*Z_train)

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

eval_mean = np.mean(X_train[:,:,0])
eval_stdev = np.std(X_train[:,:,0])
time_mean = np.mean(X_train[:,:,1])
time_stdev = np.std(X_train[:,:,1])

#for i in range(X_train.shape[2]):
#    X_train[:,:,i] = (X_train[:,:,i] - eval_mean) / eval_stdev
#    X_train[:,:,i] = (X_train[:,:,i] - time_mean) / time_stdev
#
#    X_test[:,:,i] = (X_test[:,:,i] - eval_mean) / eval_stdev
#    X_test[:,:,i] = (X_test[:,:,i] - time_mean) / time_stdev
#
#    X_dev[:,:,i] = (X_dev[:,:,i] - eval_mean) / eval_stdev
#    X_dev[:,:,i] = (X_dev[:,:,i] - time_mean) / time_stdev
#    
#    X_scaling = [eval_mean, eval_stdev, time_mean, time_stdev]

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
np.savetxt(dir_path + '/Y_scaling_info.dat', np.array([Ymean, Ystdev, Ymax, Ymin, Ycenter]))

###############################################################################
#
# Train and test model
#
###############################################################################

model1 = Sequential([
     InputLayer((min_length, X_train.shape[2])),
     BatchNormalization(),
     LSTM(64, return_sequences=True, activation = 'relu', use_bias = True, recurrent_activation='sigmoid'),
     BatchNormalization(),
     LSTM(64, return_sequences=True, activation = 'relu', use_bias = True, recurrent_activation='sigmoid'),
     BatchNormalization(),
     LSTM(64, return_sequences=False, activation = 'relu', use_bias = True, recurrent_activation='sigmoid'),
     BatchNormalization(),
     Dense(8, 'relu'),
     BatchNormalization(),
     Dense(1, 'tanh')
]);

model1.summary()
cp = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=5e-5), metrics=[RootMeanSquaredError()])
RMS_error_test_list = []
RMS_error_train_list = []

for i in range(1,1000):
    N_epochs = min(5*i,30)
    model_fit = model1.fit(X_train, Y_train, validation_data=(X_dev,Y_dev), epochs=N_epochs, callbacks=[cp], batch_size = 512)
    loss_val = model_fit.history['loss'][0]
    test_predictions = np.array(model1.predict(X_test).flatten())
    RMS_error_test, MSE_error_test = calc_error(test_predictions, Y_test)
    RMS_error_test_list.append(RMS_error_test)
    RMS_error_train_list.append(loss_val)
    np.savetxt(dir_path + '/test_error.dat', RMS_error_test_list)
    np.savetxt(dir_path + '/train_error.dat', RMS_error_train_list)
    plt.plot(Y_test[0:250], test_predictions[0:250], marker = 'o', ls = 'None', color = 'k')
    plt.plot(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000), ls = '--', color = 'r')
    plt.savefig(dir_path + '/test_data_i={}.pdf'.format(i))
    plt.close()
