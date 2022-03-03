#!/usr/bin/env python

# Import libraries
import torch.cuda

from helper_code import *
from PIL import Image
import numpy as np
import os
import pretrain_data_utils
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import cm
from wavelets_pytorch.transform import WaveletTransformTorch

DATADIR = '../datasets/transfer-learning/'

fs = 1000

# get X and y data
X, y = pretrain_data_utils.get_pretrain_data(DATADIR, fs=fs, seg_len=10)

# split into train and val sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

IMDIR = '../datasets/pretrain_img/train/'
pretrain_data_utils.save_images(X_train, y_train, im_dir=IMDIR)

IMDIR = '../datasets/pretrain_img/val/'
pretrain_data_utils.save_images(X_test, y_test, im_dir=IMDIR)

print("All data:")
print(Counter(y[:, 0]))
print("Train data:")
print(Counter(y_train[:, 0]))
print("Test data:")
print(Counter(y_test[:, 0]))





































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