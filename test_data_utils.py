from helper_code import *
import numpy as np
import preprocess_utils
from preprocess_utils import preprocess
from matplotlib import cm
from PIL import Image
from wavelets_pytorch.transform import WaveletTransformTorch

def get_test_data(data, current_recordings, verbose, fs_resample=1000, fs=4000):

    classes = ['Present', 'Unknown', 'Absent']
    n_classes = len(classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the test data...')

    processed_recordings = list()
    labels = list()
    rec_names = list()

    # Load the current patient data and recordings.
    current_patient_data = data

    # get current recording names
    current_recording_names = preprocess_utils.load_recording_names(current_patient_data)
    rec_names.append(np.vstack(current_recording_names))

    # append processed recording from each location for this patient
    n_recordings = len(current_recordings)
    for r in range(n_recordings):
        recording = preprocess(current_recordings[r], fs_resample=fs_resample, fs=fs)
        processed_recordings.append(recording)

    # Get label for each recording - this is where should differ if I want to relabel
    # for the different types of murmurs, or for depending on whether murmur heard at this
    # location or not
    # Extract labels and use one-hot encoding.
    current_labels = np.zeros(n_classes, dtype=int)
    label = get_label(current_patient_data)
    if label in classes:
        j = classes.index(label)
        current_labels[j] = 1
    labels.append(np.tile(current_labels, (n_recordings, 1)))

    rec_names = np.vstack(rec_names)
    labels = np.vstack(labels)

    return processed_recordings, labels, rec_names


def segment_test_data(X):
    fs = 1000
    seg_len = 7.5

    n_recordings = len(X)
    n_samples = int(fs*seg_len)

    X_seg = list()

    for i in range(n_recordings):
        # load current patient data
        current_recording = X[i]

        n_segs = len(current_recording) // n_samples

        start_idx = 0
        end_idx = n_samples
        X_recording = np.ndarray((n_segs, n_samples))

        for seg in range(n_segs):
            tmp = X[i][start_idx:end_idx]
            tmp = (tmp - np.mean(tmp)) / np.std(tmp)
            # check concatenation
            X_recording[seg, :] = tmp

            start_idx += n_samples
            end_idx += n_samples

        # recording too short, pad with zeros
        if n_segs == 0:
            X_recording = np.ndarray((1, n_samples))
            tmp = X[i]
            tmp = (tmp - np.mean(tmp)) / np.std(tmp)
            N = n_samples - len(tmp)
            X_recording[0, :] = np.concatenate([tmp, np.zeros(N)])

        # append segmented recordings
        X_seg.append(X_recording)

    X_seg = np.vstack(X_seg)

    return X_seg


def create_cwt_images(X):
    fs = 1000

    n_samples = X.shape[0]

    dt = 1 / fs
    dj = 1 / 10
    fb = WaveletTransformTorch(dt, dj, cuda=True)
    batch_size = 64
    batch_start = 0
    batch_end = batch_size

    cwt_imgs = list()

    while batch_end < n_samples:
        # get segment data
        data = X[batch_start:batch_end, :]

        # apply CWT to data, take abs of coeffs
        cfs = fb.cwt(data)
        cfs = abs(cfs)

        for idx, cf in enumerate(cfs):
            # save cf as image here
            img = get_cfs_as_jpg(cf)
            cwt_imgs.append(img)

        batch_start += batch_size
        batch_end += batch_size

    # apply CWT to remaining data
    data = X[batch_start:n_samples, :]
    cfs = fb.cwt(data)
    cfs = abs(cfs)

    for idx, cf in enumerate(cfs):
        # jpg image
        img = get_cfs_as_jpg(cf)
        cwt_imgs.append(img)

    return cwt_imgs

def get_cfs_as_jpg(cfs):

    # rescale cfs to interval [0, 1]
    cfs = (cfs - cfs.min()) / (cfs.max() - cfs.min())

    # create colormap
    cmap = cm.get_cmap('jet', 256)

    # apply colormap to data, return as ints from 0 to 255
    img = cmap(cfs, bytes=True)

    # convert from rgba to rgb
    img = np.delete(img, 3, 2)

    # create image from numpy array
    img = Image.fromarray(img)

    # resize the image
    img = img.resize((224, 224))

    return img