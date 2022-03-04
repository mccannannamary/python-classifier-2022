import matplotlib.pyplot as plt

import pretrain_team_code
import preprocess_utils
from wavelets_pytorch.transform import WaveletTransformTorch
import numpy as np
from matplotlib import cm
from PIL import Image
import os


def get_pretrain_data(DATADIR, fs=1000, seg_len=10):
    # Find data files
    pretrain_files = pretrain_team_code.find_pretrain_files(DATADIR)
    n_pretrain_files = len(pretrain_files)

    classes = ['Normal', 'Abnormal']
    num_classes = len(classes)

    X = list()
    y = list()
    idx = list()

    n_samples = fs * seg_len
    for i in range(n_pretrain_files):
        # load current patient data
        current_pretrain_data = pretrain_team_code.load_pretrain_data(pretrain_files[i])
        current_recording = pretrain_team_code.load_pretrain_recordings(DATADIR, current_pretrain_data)
        current_idx = pretrain_team_code.get_current_idx(DATADIR, current_pretrain_data)

        # preprocess signal
        current_recording = preprocess_utils.preprocess(current_recording, fs, 2000)

        # divide signal into segments according to n_samples in seg_len
        n_segments = len(current_recording) // n_samples

        X_recording = np.reshape(current_recording[0:n_segments * n_samples],
                                 (n_segments, n_samples))
        X.append(X_recording)

        # Extract labels and use one-hot encoding
        current_labels = np.zeros(num_classes, dtype=int)
        label = pretrain_team_code.get_pretrain_label(current_pretrain_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
            y.append(np.tile(current_labels, (n_segments, 1)))

        idx.append(np.tile(current_idx, (n_segments, 1)))

    X = np.vstack(X)
    y = np.vstack(y)
    idx = np.vstack(idx)

    # standardize each segment
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

    return X, y, idx


def save_images(X, y, im_dir, fs=1000, ):
    classes = ['normal', 'abnormal']
    n_segments = X.shape[0]

    # set up CWT filterbank
    dt = 1 / fs
    dj = 1 / 10
    fb = WaveletTransformTorch(dt, dj, cuda=True)

    for k in range(n_segments):
        # get segment data and label
        data = X[k, :]
        label = classes[np.argmax(y[k, :])]

        # apply CWT to data, take abs of coeffs
        cfs = fb.cwt(data)
        cfs = abs(cfs)

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

        # save image in appropriate class directory
        save_dir = os.path.join(im_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, str(k).zfill(4))
        img.save(fname + '.jpg')


def plot_images(X, y, fs=1000):
    classes = ['normal', 'abnormal']

    dt = 1 / fs
    dj = 1 / 10
    fb = WaveletTransformTorch(dt, dj, cuda=True)

    for k in range(len(y)):
        data = X[k, :]
        label = classes[np.argmax(y[k, :])]

        cfs = fb.cwt(data)
        cfs = abs(cfs)

        t = np.linspace(0, len(data) / fs, len(data))
        fig, ax = plt.subplots(2, 1, figsize=(12,8))
        ax = ax.flatten()
        ax[0].plot(t, data, linewidth=0.5)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_title(label)

        T, F = np.meshgrid(t, fb.fourier_frequencies)

        cf = ax[1].contourf(T, F, cfs, 100)
        fig.colorbar(cf, ax=ax[1])
        ax[1].set_title('scalogram')

        plt.tight_layout()
        plt.show()
