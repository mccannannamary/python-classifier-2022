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
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
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
from pretrain_model_utils import ImageFolderWithNames


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):

    create_dataset = True

    pretrained_model_folder = './pretrain_seg_alexnet/'

    data_folders = [data_folder, '../datasets/circor/val/']
    image_folders = ['../datasets/seg_images/train/', '../datasets/seg_images/val/']
    pt_ids_names = ['pt_ids_train', 'pt_ids_val']

    if create_dataset:
        for i, data_folder in enumerate(data_folders):

            # Find data files.
            if verbose >= 1:
                print('Finding data files...')

            recordings, features, labels, rec_names, pt_ids = preprocess_utils.get_challenge_data(data_folder, verbose, fs_resample=1000, fs=2000)

            # now perform segmentation
            X_seg, y_seg, names_seg = preprocess_utils.segment_challenge_data(recordings, labels, rec_names)

            # save patient ids
            im_dir = '../datasets/circor_img_seg/'
            os.makedirs(im_dir, exist_ok=True)
            fname = os.path.join(im_dir, pt_ids_names[i])
            np.save(fname, pt_ids)

            # now create and save a CWT image for each segmented FHS
            preprocess_utils.create_cwt_images(X_seg, y_seg, names_seg, image_folders[i])

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    train_set = datasets.ImageFolder(root=image_folders[0], transform=transform)
    valid_set = datasets.ImageFolder(root=image_folders[1], transform=transform)

    # Create a torch.device() which should be the GPU if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 0.001
    batch_size = 15
    # classifier = NeuralNetClassifier(
    #     module=AlexNet(n_hidden_units=256, n_classes=3),
    #     criterion=nn.CrossEntropyLoss,
    #     lr=0.001,
    #     batch_size=batch_size,
    #     max_epochs=15,
    #     optimizer=optim.SGD,
    #     optimizer__momentum=0.9,
    #     optimizer__nesterov=False,
    #     optimizer__weight_decay=0.01,
    #     train_split=predefined_split(valid_set),
    #     iterator_train__shuffle=True,
    #     iterator_train__num_workers=8,
    #     iterator_valid__shuffle=False,
    #     iterator_valid__num_workers=8,
    #     callbacks=[
    #         ('lr_scheduler',
    #          LRScheduler(policy='ReduceLROnPlateau')),
    #         ('checkpoint',
    #          Checkpoint(dirname=model_folder,
    #                     monitor='valid_acc_best',
    #                     f_pickle='best_model.pkl'))
    #     ],
    #     device=device
    # ).fit(train_set, y=None)

    model = load_challenge_model(pretrained_model_folder, verbose)
    classifier = model['classifier']

    classifier.fit(train_set, y=None)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    classes = ['absent', 'present', 'unknown']
    save_challenge_model(model_folder, classes, classifier)

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
    classifier = model['classifier']

    np.seterr(all='raise')

    # need to do whole process of filtering, segmenting, and getting FHS images here
    X_fhs = test_data_utils.segment_data(recordings)

    # create CWT image for each segmented FHS - need to figure out how to
    # return array of images, or else create a directory with the images
    # and load one by one (but should be able to store array of images)
    test_imgs = test_data_utils.create_cwt_images(X_fhs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    img_t = list()
    for img in test_imgs:
        img_t.append(transform(img).unsqueeze(0))
    img_t = torch.vstack(img_t)

    # run each image through model
    img_probabilities = classifier.predict_proba(img_t)

#    img_probabilities = np.ndarray((len(test_imgs), len(classes)))
#    for k, img in enumerate(test_imgs):
#        img_t = transform(img).unsqueeze(0)
#        img_probabilities[k, :] = classifier.predict_proba(img_t)

    # tmp_probabilities = np.mean(img_probabilities, axis=0)
    # probabilities = np.empty_like(tmp_probabilities)
    # probabilities[0] = tmp_probabilities[1]
    # probabilities[1] = tmp_probabilities[2]
    # probabilities[2] = tmp_probabilities[0]
    probabilities = np.mean(img_probabilities, axis=0)

    # Get classifier probabilities.
    #probabilities = classifier.predict_proba(features)
    #probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    idx = np.argmax(probabilities)
    labels[idx] = 1

    return classes, labels, probabilities


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, classes, classifier):
    d = {'classes': classes, 'classifier': classifier}
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