#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
import torchvision.models

import preprocess_utils
import test_data_utils
from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, joblib
import glob, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from skorch.callbacks import Checkpoint
from pretrain_model_utils import ResNet18
from sklearn.model_selection import train_test_split


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):

    split_dataset = True
    create_dataset = True

    # do stratified split of all available data into train, validation, and test folders
    # to prevent overfitting when training model
    train_folder = './datasets/pt_files/train/'
    val_folder = './datasets/pt_files/val/'

    if split_dataset:
        patient_files = find_patient_files(data_folder)
        n_patient_files = len(patient_files)

        classes = ['Present', 'Unknown', 'Absent']
        n_classes = len(classes)

        pt_ids = list()
        labels = list()

        for i in range(n_patient_files):
            current_patient_data = load_patient_data(patient_files[i])
            current_patient_id = current_patient_data.split('\n')[0].split(' ')[0]
            pt_ids.append(current_patient_id)

            current_labels = np.zeros(n_classes, dtype=int)
            label = get_label(current_patient_data)
            if label in classes:
                j = classes.index(label)
                current_labels[j] = 1
            labels.append(current_labels)

        # perform stratified random split by labels
        ids_train, ids_val, labels_train, labels_val = \
            train_test_split(pt_ids,
                             labels,
                             test_size=0.2,
                             random_state=1,
                             stratify=labels)

        # get all files matching ids_train and move to train folder (glob and shutils)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        for pt_id in ids_train:
            tmp = os.path.join(data_folder, pt_id + '*')
            for file in glob.glob(tmp):
                shutil.copy(file, train_folder)

        for pt_id in ids_val:
            tmp = os.path.join(data_folder, pt_id + '*')
            for file in glob.glob(tmp):
                shutil.copy(file, val_folder)

    data_folders = [train_folder, val_folder]
    image_folders = ['./datasets/cwt_imgs/train/',
                     './datasets/cwt_imgs/val/']
    image_relabel_folders = ['./datasets/relabel_cwt_imgs/train/',
                             './datasets/relabel_cwt_imgs/val/']

    # using split dataset, create CWT images from segments of PCG data and save in 'image_folders'
    if create_dataset:
        for i, data_folder in enumerate(data_folders):

            # Find data files.
            if verbose >= 1:
                print('Finding data files...')

            recordings, features, labels, relabels, rec_names = \
                preprocess_utils.get_challenge_data(data_folder, verbose, fs_resample=1000, fs=4000)

            # now perform segmentation
            X, y, y_relabel, names_seg = \
                preprocess_utils.segment_challenge_data(recordings, labels, relabels, rec_names)

            # now create and save a CWT image for each PCG segment
            preprocess_utils.create_cwt_images(X, y, y_relabel, names_seg, image_folders[i], image_relabel_folders[i])

    # Train neural net.
    if verbose >= 1:
        print('Training neural network...')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # create pytorch datasets from folders where we saved images
    train_set = datasets.ImageFolder(root=image_folders[0], transform=data_transforms['train'])
    valid_set = datasets.ImageFolder(root=image_folders[1], transform=data_transforms['val'])

    # Create a torch.device() which should be the GPU if CUDA is available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    net = NeuralNetClassifier(
        module=ResNet18(n_classes=2),
        criterion=nn.CrossEntropyLoss,
        lr=0.001,
        batch_size=4,
        max_epochs=30,
        optimizer=optim.SGD,
        optimizer__momentum=0.9,
        optimizer__weight_decay=0.0005,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        iterator_train__num_workers=8,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=8,
        callbacks=[
            ('lr_scheduler',
             LRScheduler(policy='StepLR', step_size=7, gamma=0.1)),
            ('checkpoint',
             Checkpoint(dirname=model_folder,
                        monitor='valid_acc_best',
                        f_params='model.pkl',
                        f_optimizer='opt.pkl',
                        f_criterion='criterion.pkl',
                        f_history='history.json')),
        ],
        device=device
    )

    # initialize neural network
    net.initialize()

    # load parameters from pretrained model
    pretrained_model_folder = './pretrain_resnet_unfreeze/'
    param_fname = os.path.join(pretrained_model_folder, 'model.pkl')
    net.load_params(f_params=param_fname)

    # change number of classes in classification layer
    n_in_features = net.module.model.fc.in_features
    net.module.model.fc = nn.Linear(in_features=n_in_features, out_features=3)

    net.fit(train_set, y=None)

    # Choose best decision rule
    classes = ['absent', 'present', 'unknown']

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_challenge_model(model_folder, classes, net)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):

    classes = model['classes']
    net = model['net']

    # preprocess test data
    recordings = test_data_utils.get_test_data(data, recordings, verbose, fs_resample=1000, fs=4000)

    # segment test data
    X_seg = test_data_utils.segment_test_data(recordings)

    # create CWT image for each PCG segment
    test_imgs = test_data_utils.create_cwt_images(X_seg)

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_t = list()
    for img in test_imgs:
        img_t.append(transform(img).unsqueeze(0))
    img_t = torch.vstack(img_t)

    # run each image through model
    img_probabilities = net.predict_proba(img_t)

    # decision rule: first check if any images are classified as unknown, if yes, then classify
    # as unknown. If no, then move on to present. If no, then classify as absent.
    abs_idx = classes.index('absent')
    pres_idx = classes.index('present')
    unknown_idx = classes.index('unknown')

    probabilities = np.mean(img_probabilities, axis=0)
    labels = np.zeros(len(classes), dtype=np.int_)
    th1 = 0.06
    th2 = 0.07
    if probabilities[pres_idx] > th1:
        idx = pres_idx
    elif probabilities[unknown_idx] > th2:
        idx = unknown_idx
    else:
        idx = abs_idx
    labels[idx] = 1

    return classes, labels, probabilities


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, classes, net):
    d = {'classes': classes, 'net': net}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)