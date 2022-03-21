#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for running models for the 2022 Challenge. You can run it as follows:
#
#   python run_model.py model data outputs
#
# where 'model' is a folder containing the your trained model, 'data' is a folder containing the Challenge data, and 'outputs' is a
# folder for saving your model's outputs.

import numpy as np, os, sys
from helper_code import *
from team_code import load_challenge_model, run_challenge_model

# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, verbose):
    # Load models.
    if verbose >= 1:
        print('Loading Challenge model...')

    model = load_challenge_model(model_folder, verbose) ### Teams: Implement this function!!!

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    #os.makedirs(output_folder, exist_ok=True)

    # Run the team's model on the Challenge data.
    if verbose >= 1:
        print('Running model on Challenge data...')

    # Iterate over the patient files.
    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        patient_data = load_patient_data(patient_files[i])
        recordings = load_recordings(data_folder, patient_data)

        # Allow or disallow the model to fail on parts of the data; helpful for debugging.
        try:
            classes, label_list, probabilities = \
                run_challenge_model(model, patient_data, recordings, verbose) ### Teams: Implement this function!!!
        except:
            if allow_failures:
                if verbose >= 2:
                    print('... failed.')
                classes, labels, probabilities = list(), list(), list()
            else:
                raise

        output_folders = []
        thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        for th1 in thresholds:
            for th2 in thresholds:
                output_folders.append(output_folder + '_' + str(th1) + '_' + str(th2))

        for k, folder in enumerate(output_folders):
            labels = label_list[k]
            # Save Challenge outputs.
            os.makedirs(folder, exist_ok=True)
            head, tail = os.path.split(patient_files[i])
            root, extension = os.path.splitext(tail)
            output_file = os.path.join(folder, root + '.csv')
            patient_id = get_patient_id(patient_data)
            save_challenge_outputs(output_file, patient_id, classes, labels, probabilities)

    if verbose >= 1:
        print('Done.')

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 4 or len(sys.argv) == 5):
        raise Exception('Include the model, data, and output folders as arguments, e.g., python run_model.py model data outputs.')

    # Define the model, data, and output folders.
    model_folder = sys.argv[1]
    data_folder = sys.argv[2]
    output_folder = sys.argv[3]

    # Allow or disallow the model to fail on parts of the data; helpful for debugging.
    allow_failures = False

    # Change the level of verbosity; helpful for debugging.
    if len(sys.argv)==5 and is_integer(sys.argv[4]):
        verbose = int(sys.argv[4])
    else:
        verbose = 1

    run_model(model_folder, data_folder, output_folder, allow_failures, verbose)
