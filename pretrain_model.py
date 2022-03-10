#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for training models for the 2022 Challenge. You can run it as follows:
#
#   python train_model.py data model
#
# where 'data' is a folder containing the Challenge data and 'model' is a folder for saving your model.

import sys
from helper_code import is_integer
from pretrain_team_code import pretrain_challenge_model

if __name__ == '__main__':
    # Parse the arguments.

    # Define the data and model folders.
    input_folder = sys.argv[1]
    model_folder = sys.argv[2]

    pretrain_challenge_model(input_folder, model_folder)
