#!/usr/bin/env python

from helper_code import *
import glob
import shutil
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.model_selection import train_test_split

DATADIR = '../training-data/training_data/'
TRAINDIR = '../datasets/circor/train/'
VALDIR = '../datasets/circor/val/'
TESTDIR = '../datasets/circor/test/'

patient_files = find_patient_files(DATADIR)
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

# stratified random split by labels
ids_train, ids_test, labels_train, labels_test = \
    train_test_split(pt_ids,
                     labels,
                     test_size=0.2,
                     random_state=1,
                     stratify=labels)

# take half of test set and use for validation
ids_val, ids_test, labels_val, labels_test = \
    train_test_split(ids_test,
                     labels_test,
                     test_size=0.5,
                     random_state=1,
                     stratify=labels_test)

# get all files matching ids_train and move to train folder (glob and shutils)
for pt_id in ids_train:
    tmp = DATADIR + pt_id + '*'
    for file in glob.glob(tmp):
        shutil.copy(file, TRAINDIR)

for pt_id in ids_val:
    tmp = DATADIR + pt_id + '*'
    for file in glob.glob(tmp):
        shutil.copy(file, VALDIR)

# get all files matching ids_test and move to test folder (glob and shutils)
for pt_id in ids_test:
    tmp = DATADIR + pt_id + '*'
    for file in glob.glob(tmp):
        shutil.copy(file, TESTDIR)
