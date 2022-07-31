#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
import preprocess_utils
import test_data_utils
from helper_code import *
import numpy as np, scipy as sp, pandas as pd, scipy.stats, os, joblib
from collections import Counter
import glob, shutil
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.model_selection import train_test_split
from preprocess_utils import train_net

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose, relabel, freeze_shallow, pretrain):

    split_dataset = False
    create_dataset = False

    # do stratified split of all available data into train, validation, and test folders
    # to prevent overfitting when training model
    train_folder = './datasets/pt_files/train/'
    val_folder = './datasets/pt_files/val/'

    if split_dataset:
        patient_files = find_patient_files(data_folder)
        n_patient_files = len(patient_files)

        murmur_classes = ['Present', 'Unknown', 'Absent']
        n_murmur_classes = len(murmur_classes)
        outcome_classes = ['Abnormal', 'Normal']
        n_outcome_classes = len(outcome_classes)

        pt_ids = list()
        murmurs = list()

        for i in range(n_patient_files):
            current_patient_data = load_patient_data(patient_files[i])
            current_patient_id = current_patient_data.split('\n')[0].split(' ')[0]
            pt_ids.append(current_patient_id)

            current_labels = np.zeros(n_murmur_classes, dtype=int)
            label = get_murmur(current_patient_data)
            if label in murmur_classes:
                j = murmur_classes.index(label)
                current_labels[j] = 1
            murmurs.append(current_labels)

        # perform stratified random split by murmurs
        ids_train, ids_val, labels_train, labels_val = \
            train_test_split(pt_ids,
                             murmurs,
                             test_size=0.2,
                             random_state=1,
                             stratify=murmurs)

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
    murmur_image_folders = ['./datasets/imgs/murmur/train/', './datasets/imgs/murmur/val/']
    murmur_image_relabel_folders = ['./datasets/imgs/relabeled_murmur/train/', './datasets/imgs/relabeled_murmur/val/']
    outcome_image_folders = ['./datasets/imgs/outcome/train/', './datasets/imgs/outcome/val/']

    # using split dataset, create CWT images from segments of PCG data and save in 'murmur_image_folders'
    if create_dataset:
        for i, data_folder in enumerate(data_folders):

            # Find data files.
            if verbose >= 1:
                print('Finding data files...')

            recordings, features, murmurs, relabeled_murmurs, outcomes, rec_names = \
                preprocess_utils.get_challenge_data(data_folder, verbose, fs_resample=1000, fs=4000)

            # now perform segmentation
            X, y_murmurs, y_relabeled_murmurs, y_outcomes_seg, names_seg = \
                preprocess_utils.segment_challenge_data(recordings, murmurs, relabeled_murmurs, outcomes, rec_names)

            # now create and save a CWT image for each PCG segment
            preprocess_utils.create_cwt_images(X, y_murmurs, y_relabeled_murmurs, y_outcomes_seg, names_seg,
                                               murmur_image_folders[i], murmur_image_relabel_folders[i],
                                               outcome_image_folders[i])


    # Train neural murmur_net.
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

    # train and save murmur classification model
    # create pytorch datasets from folders where we saved images
    if not relabel:
        train_set = datasets.ImageFolder(root=murmur_image_folders[0], transform=data_transforms['train'])
        valid_set = datasets.ImageFolder(root=murmur_image_folders[1], transform=data_transforms['val'])
        train_classes = [label for _, label in train_set]
        class_count = Counter(train_classes)
        class_weights = torch.Tensor([len(train_classes)/c for c in pd.Series(class_count).sort_index().values])
        murmur_net = train_net(train_set, valid_set, class_weights, scratch_name='murmur', freeze_shallow=freeze_shallow, pretrain=pretrain)

    elif relabel:
        # train and save relabeled murmur classification net
        train_set = datasets.ImageFolder(root=murmur_image_relabel_folders[0], transform=data_transforms['train'])
        valid_set = datasets.ImageFolder(root=murmur_image_relabel_folders[1], transform=data_transforms['val'])
        train_classes = [label for _, label in train_set]
        class_count = Counter(train_classes)
        class_weights = torch.Tensor([len(train_classes)/c for c in pd.Series(class_count).sort_index().values])
        murmur_relabel_net = train_net(train_set, valid_set, class_weights, scratch_name='murmur_relabel', freeze_shallow=freeze_shallow, pretrain=pretrain)

    # train outcome classification net
    train_set = datasets.ImageFolder(root=outcome_image_folders[0], transform=data_transforms['train'])
    valid_set = datasets.ImageFolder(root=outcome_image_folders[1], transform=data_transforms['val'])
    train_classes = [label for _, label in train_set]
    class_count = Counter(train_classes)
    class_weights = torch.Tensor([len(train_classes)/c for c in pd.Series(class_count).sort_index().values])
    outcome_net = train_net(train_set, valid_set, class_weights, scratch_name='outcome', freeze_shallow=freeze_shallow, pretrain=pretrain)

    # Create a folder for the model if it does not already exist.
    model_folder = model_folder + 'exp_' + str(relabel) + str(freeze_shallow) + str(pretrain)
    os.makedirs(model_folder, exist_ok=True)
    relabel_model_folder = model_folder + '_relabel'
    os.makedirs(relabel_model_folder, exist_ok=True)

    # Save the model.
    murmur_classes = ['absent', 'present', 'unknown']
    outcome_classes = ['abnormal', 'normal']
    if not relabel:
        save_challenge_model(model_folder, murmur_classes, murmur_net, outcome_classes, outcome_net)
    elif relabel:
        save_challenge_model(relabel_model_folder, murmur_classes, murmur_relabel_net, outcome_classes, outcome_net)

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

    murmur_classes = model['murmur_classes']
    murmur_net = model['murmur_net']
    outcome_classes = model['outcome_classes']
    outcome_net = model['outcome_net']

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

    # get classifier murmur_probabilities
    murmur_probabilities = murmur_net.predict_proba(img_t)
    outcome_probabilities = outcome_net.predict_proba(img_t)

    # decision rule: first check if any images are classified as unknown, if yes, then classify
    # as unknown. If no, then move on to present. If no, then classify as absent.
    abs_idx = murmur_classes.index('absent')
    pres_idx = murmur_classes.index('present')
    unknown_idx = murmur_classes.index('unknown')

    murmur_probabilities = np.mean(murmur_probabilities, axis=0)
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    th1 = 0.06
    th2 = 0.07
    if murmur_probabilities[pres_idx] > th1:
        idx = pres_idx
    elif murmur_probabilities[unknown_idx] > th2:
        idx = unknown_idx
    else:
        idx = abs_idx
    murmur_labels[idx] = 1

    # choose outcome label with highest probability
    outcome_probabilities = np.mean(outcome_probabilities, axis=0)
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # concatenate classes, labels, and probabilities
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, murmur_classes, murmur_net, outcome_classes, outcome_net):
    d = {'murmur_classes': murmur_classes,
         'murmur_net': murmur_net,
         'outcome_classes': outcome_classes,
         'outcome_net': outcome_net}
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