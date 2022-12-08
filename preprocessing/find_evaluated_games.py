'''
Author: Robert Bennett
Script prepared for Stanford CS 230 final project 

Preprocessing script to sort lichess database files into sorted files to make
further preprocessing easier. 

USAGE: find_evaluated_games.py [lichess database file name] [start number]
Use 00 as the start number if running this script on the entire database file
at once. See readme file for how to more efficiently break this up into parts
and run scripts in parallel; in this case, part will correspond to individual
fragments for files. 

Creates a subdirectory called "raw games" containing individual files for each
time control.
'''


################################################################################
#
# Import modules and build main function
#
###############################################################################

import numpy as np
import pgn
import sys
import os
import chess.pgn
import chess.engine
import json
import ast
import time

def read_and_write_evaluated(read_filename, write_filename, counter = 0):
    '''
    Parses a raw chess file and extracts all games that contain an engine
    evaluation. Write these games to a new file.

    INPUTS: 
        read_filename - name of the raw file we're working with
        write_filename - name of file to write
        counter - the value the counter should start at. Leave as 0 if running
        for the first time.
    '''
    with open(read_filename, encoding = 'utf-8') as h:
        while True:
            # Check if our batch folder exists alrleady. If it doesn't, make one.
            # If it does exist, ask the user before proceeding.
            temp_list = []
            game = chess.pgn.read_game(h)
            if game is None:
                break

            for node in game.mainline():
                comment = node.comment
                broken = comment.split('\n')
                adjusted = broken[0].split(' ')[1][0:-1]
                if ':' in adjusted:
                    flag = False
                else:
                    flag = True

                if flag:
                    counter += 1
                    with open(write_filename, 'a') as file:
                         file.write(str(game))
                         file.write('\n')
                         file.write('\n')
                break
    return(counter)


################################################################################
#
# Initialization
#
################################################################################

dir_path = os.path.dirname(os.path.abspath(__file__))
batchsize = 10**6
batchnum = -1
batchstr = '000'
current_time = time.time()
batchnums = ['001', '002', '003', '004', '005', '006', '007', '008', '009']
base_write_filename = dir_path + '/{}_{}_evals.dat'
counter = 0
max_games_per_file = 10**6
start = time.time()
part = sys.argv[1]
TimeControl = sys.argv[2]
file_counter = int(part)

################################################################################
#
# Game processing
#
################################################################################

for batchnum in batchnums:
    # make a new file if we've exceeded the max number of games per file.
    # if so, print state information.
    if counter > max_games_per_file:
        file_counter += 1
        print('Counter = {}, file counter updated to {}'.format(counter, file_counter))
        counter = 0
    
    write_filename = base_write_filename.format(TimeControl, file_counter) 
    read_filename = dir_path + '/games/raw/chess_partname={}/batchnum={}/raw_games_{}.dat'.format(part, batchnum, TimeControl)

    # call the above function on our current file to find evaluated games only. 
    counter = read_and_write_evaluated(read_filename, write_filename, counter)
    elapsed = time.time() - start
    print('Finished part {}, batch {}. Current time elapsed: {} s'.format(part, batchnum, elapsed))

                
                

            




