#!/usr/bin/env python

# Import libraries
import torch.cuda

from helper_code import *
from PIL import Image
import numpy as np
import os
import pretrain_data_utils
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
from collections import Counter
from matplotlib import cm
from wavelets_pytorch.transform import WaveletTransformTorch

DATADIR = '../datasets/transfer-learning/'

fs = 1000
seg_len = 1.5
keep_percent = 0.1
rescale_size = 64

# get X and y data
X, y, names = pretrain_data_utils.get_pretrain_data(DATADIR, fs=fs)

n_samples = int(keep_percent*len(X))
idx = sample_without_replacement(len(X), n_samples, random_state=1)

X = [X[i] for i in idx]
y = [y[i] for i in idx]
names = [names[i] for i in idx]

# split into train and val sets
X_train, X_test, y_train, y_test, idx_train, idx_test = \
    train_test_split(X, y, names, test_size=0.2, random_state=1, stratify=y)

X_train, y_train, idx_train = pretrain_data_utils.segment_pretrain_data(X_train, y_train, idx_train, fs=fs, seg_len=seg_len)
X_test, y_test, idx_test = pretrain_data_utils.segment_pretrain_data(X_test, y_test, idx_test, fs=fs, seg_len=seg_len)

IMDIR = '../datasets/pretrain_img_fhs_1.5/'
os.makedirs(IMDIR, exist_ok=True)
fname = os.path.join(IMDIR, 'idx_train')
np.save(fname, idx_train)
fname = os.path.join(IMDIR, 'idx_test')
np.save(fname, idx_test)

IMDIR = '../datasets/pretrain_img_fhs_1.5/train/'
X_cwt_train = pretrain_data_utils.create_cwt_images(X_train, y_train, idx_train, rescale_size=rescale_size, jpg_dir=IMDIR, fs=fs)
#pretrain_data_utils.save_images(X_train, y_train, idx_train, im_dir=IMDIR, fs=fs)

IMDIR = '../datasets/pretrain_img_fhs_1.5/val/'
X_cwt_test = pretrain_data_utils.create_cwt_images(X_test, y_test, idx_test, rescale_size=rescale_size, jpg_dir=IMDIR, fs=fs)
#pretrain_data_utils.save_images(X_test, y_test, idx_test, im_dir=IMDIR, fs=fs)

SAVEDIR = '../datasets/numpy_fhs_1.5/'
os.makedirs(SAVEDIR, exist_ok=True)
fname = os.path.join(SAVEDIR, 'X_train')
np.save(fname, X_cwt_train)
fname = os.path.join(SAVEDIR, 'y_train')
np.save(fname, y_train)

fname = os.path.join(SAVEDIR, 'X_test')
np.save(fname, X_cwt_test)
fname = os.path.join(SAVEDIR, 'y_test')
np.save(fname, y_test)

# print("All data:")
# print(Counter(y[:, 0]))
# print("Train data:")
# print(Counter(y_train[:, 0]))
# print("Test data:")
# print(Counter(y_test[:, 0]))

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