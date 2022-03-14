#!/usr/bin/env python

# Import libraries
import os, joblib
from helper_code import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from skorch.callbacks import Checkpoint
from pretrain_model_utils import AlexNet


def pretrain_challenge_model(data_folder, model_folder):
    os.makedirs(model_folder, exist_ok=True)

    train_dir = os.path.join(data_folder, 'train')
    val_dir = os.path.join(data_folder, 'val')
    batch_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_set = datasets.ImageFolder(root=val_dir, transform=transform)

    # Create a torch.device() which should be the GPU if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNetClassifier(
        module=AlexNet(n_hidden_units=256, n_classes=2),
        criterion=nn.CrossEntropyLoss,
        lr=0.001,
        batch_size=batch_size,
        max_epochs=15,
        optimizer=optim.SGD,
        optimizer__momentum=0.9,
        optimizer__nesterov=False,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        iterator_train__num_workers=8,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=8,
        callbacks=[
            ('lr_scheduler',
             LRScheduler(policy='ReduceLROnPlateau')),
            ('checkpoint',
             Checkpoint(dirname=model_folder,
                        monitor='valid_acc_best',
                        f_pickle='best_model_batch_15.pkl'))
        ],
        device=device
    ).fit(train_set, y=None)

    save_pretrain_model(model_folder, classes, model)

###########################################################################

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


def get_current_idx(data_folder, data):
    # split data at new lines and get file name
    recording_information = data.split('\n')[0]
    entry = recording_information.split(' ')[0]
    return entry


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

# Save your trained model.
def save_pretrain_model(model_folder, classes, classifier):
    d = {'classes': classes, 'classifier': classifier}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
