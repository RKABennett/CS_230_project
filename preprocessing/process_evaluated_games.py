'''
Author: Robert Bennett
Script prepared for Stanford CS 230 final project 

Preprocessing script to format PGN files into a custom format more convenient 
for our NN.

USAGE: process_evaluated_games.py [part number]

Use 00 as the start number if running this script on the entire database file
at once. See readme file for how to more efficiently break this up into parts
and run scripts in parallel; in this case, part will correspond to individual
fragments for files. 

Creates output files in the main directory containing the processed games.
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

def read_and_write_processed(read_filename, write_filename):
    '''
    Main function that reads games and processes them into a nice format for 
    our NN before writing them to file.

    '''
    counter = 0
    start = time.time()
    with open(read_filename, encoding = 'utf-8') as h:
        while True:
            counter += 1
            # Check if our batch folder exists alrleady. If it doesn't, make one.
            # If it does exist, ask the user before proceeding.
            temp_list = []
            game = chess.pgn.read_game(h)

            # break if we're at the end of our game file
            if game is None:
                break
            
            # empty lists that will be populated
            move_list = []
            clock_list = []
            score_list = []
            board = game.board()

            try:
                # proccess the game in a try statement so if there is an error
                # in the game we just skip it, instead of the entire script
                # crashing. 
                clock_time = 600
                for move in game.mainline_moves():
                    # populate move_list with every move
                    move_list.append(board.san(move))
                    board.push(move)


                for node in game.mainline():
                    if node.is_end():
                        # append final quantities and exit if we're at the end
                        clock_time = node.clock()
                        clock_list.append(clock_time)
                        break

                    # find relevant quantities for each move
                    white_elo = game.headers["WhiteElo"]
                    black_elo = game.headers["BlackElo"]
                    elos = [white_elo, black_elo]
                    clock_time = node.clock()
                    score = node.eval().white()
                    clock_list.append(clock_time)
                    
                    # assign checkmate as an evaluation of 9999
                    current_score = score.score(mate_score=9999)
                    score_list.append(current_score)
                    
                # find how long each player took to make each move
                white_time = 600
                black_time = 600
                new_clock_list = []
                for i in range(len(clock_list)):
                    if i % 2 == 0:
                        white_time_old = white_time
                        white_time = clock_list[i]
                        time_spent = white_time_old - white_time
                        new_clock_list.append(time_spent)
                    else:
                        black_time_old = black_time
                        black_time = clock_list[i]                                   
                        time_spent = black_time_old - black_time                
                        new_clock_list.append(time_spent) 
                
                # write all data to file for this game
                with open(write_filename, 'a') as file:
                    file.write(str(game.headers))
                    file.write('\n')
                    file.write(str(elos))
                    file.write('\n')
                    file.write(str(score_list))
                    file.write('\n')
                    file.write(str(new_clock_list))
                    file.write('\n')
                    file.write(str(move_list))
                    file.write('\n')

            except:
                print('Skipping game number {}.'.format(counter))

part = sys.argv[1]                                   

dir_path = os.path.dirname(os.path.abspath(__file__))
read_filename_base = dir_path + '/games/evals/600+0_{}_evals.dat' 
read_filename =  read_filename_base.format(part)
write_filename = dir_path + '/2022-06_600+0_{}_evals_processed.dat'.format(part) 
read_and_write_processed(read_filename, write_filename)


