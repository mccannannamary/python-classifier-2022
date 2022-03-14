from helper_code import *
import numpy as np
from plot_utils import plot_segmentations
import preprocess_utils
import hsmm_utils
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from wavelets_pytorch.transform import WaveletTransformTorch

def segment_data(X):
    fs = 1000
    seg_len = 10.0

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
            # get entire cardiac cycle
            tmp = X[i][start_idx:end_idx]
            tmp = (tmp - np.mean(tmp)) / np.std(tmp)
            # if FHS too short, pad with zeros, else cut off end
            X_recording[seg, :] = tmp

        # append segmented recordings and labels
        X_seg.append(X_recording)

    X_seg = np.vstack(X_seg)

    return X_seg


def segment_fhs(X):
    fs = 1000
    seg_len = 1.0

    # Find data files
    n_test_files = len(X)
    n_samples = int(fs*seg_len)

    X_fhs = list()

    for i in range(n_test_files):
        # load current patient data
        current_recording = X[i]

        #plt.plot(current_recording)
        #plt.show()

        lpf_recording = preprocess_utils.butterworth_low_pass_filter(current_recording, 5, 150, fs)
        assigned_states = hsmm_utils.segment(lpf_recording, fs=fs)
        #plot_segmentations(lpf_recording, assigned_states)

        idx_states = hsmm_utils.get_states(assigned_states)

        n_fhs = len(idx_states)-1

        X_recording = np.ndarray((n_fhs, n_samples))

        for row in range(n_fhs):
            # get entire cardiac cycle
            tmp = X[i][idx_states[row, 0]:idx_states[row+1, 0]]
            tmp = (tmp - np.mean(tmp)) / np.std(tmp)
            # if FHS too short, pad with zeros, else cut off end
            if len(tmp) < n_samples:
                # figure out how many samples need to be padded
                N = n_samples - len(tmp)
                X_recording[row, :] = np.concatenate((tmp, np.zeros(N)))
            else:
                X_recording[row, :] = tmp[0:n_samples]

        # append segmented recordings and labels
        X_fhs.append(X_recording)

    X_fhs = np.vstack(X_fhs)

    return X_fhs

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