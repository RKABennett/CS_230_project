'''
Helper functions for the NN used to guess chess ratings.
Written by Robert Bennett for the CS 230 final project.
'''

###############################################################################
#
# Import functions
#
###############################################################################

import os
import numpy as np
import ast
import time
import matplotlib.pyplot as plt
import chess_NN_helper_funcs as cf
import tensorflow as tf                                                         
import pandas as pd                                                             
from tensorflow.keras.models import Sequential                                  
from tensorflow.keras.layers  import *                                          
from tensorflow.keras.callbacks import ModelCheckpoint                          
from tensorflow.keras.losses import MeanSquaredError                            
from tensorflow.keras.metrics import RootMeanSquaredError                       
from tensorflow.keras.optimizers import Adam                                    
import scipy.stats
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

# global variables to be called by helper functions
move_dictionary = dict()
square_dictionary = dict()
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8']

for i in range(8):
    for j in range(0,8):
        square_l = letters[i]
        square_n = numbers[j]
        square_dictionary[square_l + square_n] = (i,j)

def calc_error(prediction, actual):
    '''
    Calculate the RMS error for predicted vs. actual values.
    '''
    error_array = prediction - actual
    mean_sq_error = np.sum(error_array**2)/(np.size(error_array))
    RMS_error = np.sqrt(np.mean(error_array**2))
    return RMS_error, mean_sq_error

def load_data(file_list, min_length = 40):
    '''
    Load the data used to train or test the NN. Here, file list is a list of
    files (including paths) to preprocesed files with stored game data.
    '''
    X = []
    Y = []
    for data_file in file_list:
        print(data_file)
        with open(data_file) as file:
            counter = 0
            for line in file:
                counter += 1
                game_matrix = []
                
                if counter == 1:
                    game_win = 0
                    game_normal = 0
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    line = line.replace('\n', '')
                    line = line.replace('\'', '')
                    line = line.split(',')

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
                    

                if counter == 2:
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    line = line.replace('\n', '')
                    line = line.replace('\'', '')
                    line = line.split(',')
                    elo = np.mean([float(i) for i in line])
                    elo_diff = float(line[0]) - float(line[1])
                elif counter == 3:
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    line = line.replace('\n', '')
                    line = line.split(',')
                    eval_list = ([float(i) for i in line]) 
                    #eval_list = np.array(eval_list)

                elif counter == 4:
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    line = line.replace('\n', '')
                    line = line.split(',')
                    time_list = ([float(i) for i in line]) 
                    #time_list = np.array(time_list)

                elif counter == 5:
                    counter = 0
                    line = ast.literal_eval(line)
                    for i, move in enumerate(line):
                        iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo,\
                        isking, isqueen, isrook, isbishop, isknight, ispawn,\
                        row, col = move2vec(move)
                        
                        try:
                            cpl = eval_list[i]
                        except:
                            cpl = -1
                        time_spent = time_list[i]
                        move_vector = np.array([cpl, time_spent, game_win,\
                        game_normal, iscapture, ischeck, iscastle_ks,\
                        iscastle_qs, ispromo, isking, isqueen, isrook,\
                        isbishop, isknight, ispawn, row, col])
                        game_matrix.append(move_vector.T)

                    if len(game_matrix) >= min_length:
                        X.append(game_matrix[:min_length])
                        Y.append(elo)

    return X,Y


def move2vec(move):
    '''
    Converts a move from the preprocessed file format to a format that can be
    handled with our NN.
    '''
    global square_dictionary
    global move_dictionary
    
    try:
        iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo, isking, isqueen, 
        isrook, isbishop, isknight, ispawn, row, col = move_dictionary[move]
        return iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo, isking, 
        isqueen, isrook, isbishop, isknight, ispawn, row, col
    except:
        pass

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
    
    filtered_move = move.replace('+', '').replace('x', '')\
                    .replace('#', '').replace('=','')
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

    move_dictionary[move] = iscapture, ischeck, iscastle_ks, iscastle_qs,\
    ispromo, isking, isqueen, isrook, isbishop, isknight, ispawn, row, col
    return iscapture, ischeck, iscastle_ks, iscastle_qs, ispromo, isking,\
    isqueen, isrook, isbishop, isknight, ispawn, row, col

def build_model(load_model_bool, X_shape, modelname = '', min_length = 40):

    if load_model_bool:
        model = load_model(modelname)
        return model
        

    # Initialize LSTM layers
    N_LSTM = 2
    tf.keras.backend.clear_session()
    m_LSTM1 = 101
    m_LSTM2 = 174
    m_LSTM = [m_LSTM1, m_LSTM2]
    LSTM_activation_func = 'tanh'
    LSTM1 = LSTM(m_LSTM1, 
                 return_sequences=True, 
                 activation = LSTM_activation_func, 
                 use_bias = True, 
                 recurrent_activation='sigmoid'
                 )
    LSTM2 = LSTM(m_LSTM2, 
                 return_sequences=False, 
                 activation = LSTM_activation_func, 
                 use_bias = True, 
                 recurrent_activation='sigmoid'
                 )
    LSTMs = [LSTM1, LSTM2]

    # Initialize dense layers
    N_dense = 4
    m_dense1 = 62
    m_dense2 = 100
    m_dense3 = 95
    m_dense4 = 15
    dense1_func = 'tanh'
    dense2_func = 'relu'
    dense3_func = 'relu'
    dense4_func = 'tanh'
    m_dense = [m_dense1, m_dense2, m_dense3, m_dense4]
    Dense1 = Dense(m_dense1, dense1_func)
    Dense2 = Dense(m_dense2, dense2_func)
    Dense3 = Dense(m_dense3, dense3_func)
    Dense4 = Dense(m_dense4, dense4_func)
    Denses = [Dense1, Dense2, Dense3, Dense4]



    # initalize model
    model = Sequential()
    model.add(InputLayer((min_length, X_shape)))
    model.add(BatchNormalization())

    # add all LSTM layers
    for l in range(N_LSTM):
        model.add(LSTMs[l])
        model.add(BatchNormalization())

    # add all dense layers
    for l in range(N_dense):
        model.add(Denses[l])
        model.add(BatchNormalization())

    # final output layers
    model.add(Dense(1, 'tanh'))

    return model

