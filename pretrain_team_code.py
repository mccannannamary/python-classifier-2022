#!/usr/bin/env python

# Import libraries

from helper_code import *
import numpy as np
import scipy as sp
import scipy.stats
import preprocess_utils
import hsmm_utils
import plot_utils
import os
import sys
import joblib
import time
from sklearn.impute import SimpleImputer

fs = 1000

def pretrain_challenge_model(input_folder, output_folder):
    # Find data files
    pretrain_files = find_pretrain_files(input_folder)
    num_pretrain_files = len(pretrain_files)

    # Create output data folder if it does not already exist
    os.makedirs(output_folder,exist_ok=True)

    classes = ['Normal', 'Abnormal']
    num_classes = len(classes)

    labels = list()
    #images = list()

    for i in range(num_pretrain_files):

        # Load current patient data
        current_pretrain_data = load_pretrain_data(pretrain_files[i])
        current_recording = load_pretrain_recordings(input_folder, current_pretrain_data)

        # preprocess signal
        current_recording = preprocess_utils.preprocess(current_recording, fs, 2000)

        # segment signal
        assigned_states = hsmm_utils.segment(current_recording, fs)

        # check segmentations
        # plot_utils.plot_segmentations(current_recording, assigned_states, fs)

        # Extract input images
        #current_image = get_image(current_pretrain_data, current_recording)
        #images.append(images)

        # Extract labels and use one-hot encoding
        current_labels = np.zeros(num_classes, dtype=int)
        label = get_pretrain_label(current_pretrain_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
            labels.append(current_labels)

    labels = np.vstack(labels)
    #images = np.vstack(images)

    x = 5

# Find pretrain data files.
def find_pretrain_files(data_folder):
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension == '.hea':
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames

# load patient pretrain data as a string.
def load_pretrain_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

def load_pretrain_recordings(data_folder, data):
    # split data at new lines and get line with .wav PCG file
    recording_information = data.split('\n')[1]
    entry = recording_information.split(' ')
    # isolate .wav file name from line
    recording_file = entry[0]
    filename = os.path.join(data_folder, recording_file)
    recording, _ = load_wav_file(filename)

    return recording

# get label from pretrain data
def get_pretrain_label(data):
    label = None
    for l in data.split('\n'):
        if l.startswith('# '):
            try:
                label = l.split(' ')[1]
            except:
                pass
    return label
