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
            X_recording[seg, :] = tmp

            start_idx += n_samples
            end_idx += n_samples

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