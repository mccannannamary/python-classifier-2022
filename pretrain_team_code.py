#!/usr/bin/env python

# Import libraries
import os
from helper_code import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from skorch.callbacks import Checkpoint
from pretrain_model_utils import PretrainedModel
from pretrain_model_utils import AlexNetMod

MODEL_DIR = './model/alexnet_fhs_2.5/'
os.makedirs(MODEL_DIR, exist_ok=True)


def pretrain_challenge_model(input_folder):
    train_dir = os.path.join(input_folder, 'train')
    val_dir = os.path.join(input_folder, 'val')
    batch_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_set = datasets.ImageFolder(root=val_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)

    # imgs = list()
    # labels = list()
    # for k, (img, label) in enumerate(train_set):
    #     imgs.append(img)
    #     labels.append(label)


    # Create a torch.device() which should be the GPU if CUDA is available,
    # otherwise use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model with two callbacks: (1) learning rate scheduler, scales
    # learning rate by gamma every step_size steps? (what is meant by steps? epochs?
    # (2) checkpoint, creates checkpoint of model after each epoch if model meets
    # "monitor" criteria (best validation accuracy or lowest validation loss)
    model = NeuralNetClassifier(
        module=PretrainedModel(n_hidden_units=256),
        criterion=nn.CrossEntropyLoss,
        lr=0.001,
        batch_size=batch_size,
        max_epochs=10,
        optimizer=optim.SGD,
        optimizer__momentum=0.9,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        iterator_train__num_workers=8,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=8,
        callbacks=[
            ('lr_scheduler',
             LRScheduler(policy='ReduceLROnPlateau')),
            ('checkpoint',
             Checkpoint(dirname=MODEL_DIR,
                        f_pickle='best_model1.pkl'))
        ],
        device=device
    )

    # set up grid search parameters
    params = {
        'batch_size': [10, 32, 64],
        'lr': [1e-4, 1e-3, 1e-2],
        'callbacks__lr_scheduler__step_size': [1, 3, 5, 7],
        'module__n_hidden_units': [128, 256, 512, 1024],
        'optimizer__nesterov': [False, True],
    }

    # gs = GridSearchCV(estimator=model, param_grid=params, refit=False, cv=3, scoring='accuracy', verbose=2)

    # gs.fit(torch.stack(imgs), y=torch.tensor(labels))

    model.fit(train_set, y=None)

    # save model
    # with open('model/alexnet_fhs/alexnet_fhs.pkl', 'wb') as f:
    #     pickle.dump(model, f)


####################################################################################

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
