'''
Author: Robert Bennett
Script prepared for Stanford CS 230 final project 

Preprocessing script to sort lichess database files into sorted files to make
further preprocessing easier.

USAGE: process_into_smaller_files.py [lichess database file name] [start number]

Use 00 as the start number if running this script on the entire database file
at once. See readme file for how to more efficiently break this up into parts
and run scripts in parallel; in this case, part will correspond to individual
fragments for files. 

Creates a subdirectory called "raw games" containing individual files for each
time control.
'''

################################################################################
#
# Import modules
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

################################################################################
#
# Initialization
#
################################################################################

dir_path = os.path.dirname(os.path.abspath(__file__)) # current directory
counter = 0

chess_filename = dir_path + '/' + sys.argv[1]
chess_partname = sys.argv[2]

# all of the time controls in the lichess databse
TimeControls = ['60+0', 
                '120+1', 
                '180+0', 
                '180+2', 
                '300+0', 
                '300+3', 
                '600+0', 
                '600+5', 
                '900+10', 
                '1800+0', 
                '1800+20'
                ]

batchsize = 10**6 # number of files per batch, suggested is 10^6 to keep things 
                  # easy to manage
batchnum = -1     # starting batch number -1
batchstr = '000'  # initial batch string

current_time = time.time()

################################################################################
#
# Process games into subfiles
#
################################################################################

with open(chess_filename, encoding = 'utf-8') as h:
    while True:
        # Check if our batch folder exists alrleady. If it doesn't, make one.
        # If it does exist, ask the user before proceeding.

        # Print progress every 5000 games for ease of monitoring
        if counter % 5000 == 0:
            old_time = current_time
            current_time = time.time()
            time_spent = current_time - old_time
            print('Iteration {}, time spent: {} s'.format(counter, time_spent))

        if counter % batchsize == 0:
            batchnum += 1 # once we reach our batch size, reset and move to the 
                          # next batch
            batchstr = str(batchnum)

            if len(batchstr) == 1:
                batchstr = '00' + batchstr
            elif len(batchstr) == 2:
                batchstr = '0' + batchstr
            if os.path.exists(dir_path + '/raw_games/chess_partname={}/batchnum={}'.format(chess_partname, batchstr)):
                prompt = input('WARNING: Folder exists! Type "continue" to process anyway.')
                if not prompt == 'continue':
                    break
            else:
                os.makedirs(dir_path + '/raw_games/chess_partname={}/batchnum={}'.format(chess_partname, batchstr))
        
        try:
            temp_list = []
            counter += 1
            game = chess.pgn.read_game(h)
            if game is None: # move outside of the for loop when the file ends
                break

            # Extract relevant information from the pgn file to write to a
            # smaller raw file
            game.headers["UTCDate"] = game.headers["UTCDate"] + '_' + str(counter)
            for TimeControl in TimeControls:
                if game.headers['TimeControl'] == TimeControl:
                    with open(dir_path + '/raw_games/chess_partname={}/batchnum={}/raw_games_{}.dat'.format(chess_partname, batchstr,TimeControl), 'a') as file:
                        file.write(str(game))
                        file.write('\n')
                        file.write('\n')

        except:
           # Rarely the database contains missing fields which breaks the 
           # script. In this case, we just skip the game altogether and tell
           # the user we're skipping it. This happens so rarely that it
           # doesn't matter (<10 times per million games) but we need to have
           # the try/except statement to avoid crashing the script. 
           print("Skipping game number {}".format(counter))

print('Part {} completed!'.format(chess_partname))

            
            

        




