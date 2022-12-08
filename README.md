Written by Robert Bennett for CS 230.

This folder contains the Python scripts used to preprocess data for the 
chess-guessing NN, the Python script used to train the NN, and a script 
container helper functions. 

    1. Chess_guess_rating_NN.py
    Main script used for building/training the NN.

    2. Chess_NN_helper_funcs.py
    Helper functions used in the main script

    3. Preprocessing directory
    Contains various files used to preprocess data. The full preprocessing
    procedure is described below.

###############################################################################
#
# PREPROCESSING PROCEDURE
#
###############################################################################

The preprocessing folder contains preprocessing scripts to work lichess database 
files into a more convenient format for the neural net. 

################################################################################
#
# Initial sorting 
#
################################################################################

Begin by downloading the main database file and moving it to this directory. 
Then, we use the following command to break it into ~10 smaller files that 
are easier to work with:

    split -C 20000m --numeric-suffixes [database filename] part

Afterwards, launch parallel_preprocess_raw to simultaneously work all 10
fragmented files into files sorted by time control. To do this, do:

    parallel -j 10 :::: parallel_preprocess_raw.sh

This will create a new subdirectory containing all of the raw files sorted by
time control.

After completing this step, you may optionally delete the main database file
and the individual part files.

################################################################################
#
# Screening for evaluated games
#
################################################################################

Next, we preprocess the raw files into only evaluated games (i.e., games that
have been evaluated using an engine and contain scores for every move; about
10% of the lichess database has these evals).

To do this in parallel, do:

    parallel -j 10 :::: parallel_find_evals.sh

This will create a new subfolder, evals, containing 10 files with ONLY
games that contain engine evaluations.

################################################################################
#
# Final formatting
#
################################################################################

Lastly, we format the evaluated game files into a non-pgn format that's more
convenient for our NN. To do this in parallel, launch:

    parallel -j 10 :::: parallel_process_games.sh 

This will output 10 files containing the games as they may be input into the NN.

