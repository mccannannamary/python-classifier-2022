import os
import matplotlib.pyplot as plt
import pretrain_team_code
import preprocess_utils
from wavelets_pytorch.transform import WaveletTransformTorch
import numpy as np
from matplotlib import cm
from PIL import Image
from skimage.transform import resize
import hsmm_utils


def get_pretrain_data(DATADIR, fs=1000):
    # Find data files
    pretrain_files = pretrain_team_code.find_pretrain_files(DATADIR)
    n_pretrain_files = len(pretrain_files)

    classes = ['Normal', 'Abnormal']
    num_classes = len(classes)

    X = list()
    y = list()
    rec_names = list()

    for i in range(n_pretrain_files):
        # load current patient data
        current_pretrain_data = pretrain_team_code.load_pretrain_data(pretrain_files[i])
        current_recording = pretrain_team_code.load_pretrain_recordings(DATADIR, current_pretrain_data)

        current_recording_name = current_pretrain_data.split('\n')[0]
        current_recording_name = current_recording_name.split(' ')[0]
        rec_names.append(current_recording_name)

        # preprocess signal
        current_recording = preprocess_utils.preprocess(current_recording, fs, 2000)

        X.append(current_recording)

        # Extract labels and use one-hot encoding
        current_labels = np.zeros(num_classes, dtype=int)
        label = pretrain_team_code.get_pretrain_label(current_pretrain_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
            y.append(current_labels)

    return X, y, rec_names


def segment_pretrain_data(X, y, rec_names, fs=1000, seg_len=7.5):
    # Find data files
    n_pretrain_files = len(X)
    n_samples = int(fs*seg_len)

    X_s = list()
    y_s = list()
    names_s = list()

    for i in range(n_pretrain_files):
        # load current patient data
        current_recording = X[i]
        current_name = rec_names[i]

        n_seg = len(current_recording) // n_samples
        X_recording = np.ndarray((n_seg, n_samples))
        start_idx = 0
        end_idx = n_samples

        for seg in range(n_seg):
            # get entire cardiac cycle
            tmp = X[i][start_idx:end_idx]
            tmp = (tmp - np.mean(tmp)) / np.std(tmp)
            X_recording[seg, :] = tmp

            names_s.append(current_name + '_' + str(seg).zfill(3))

            start_idx += n_samples
            end_idx += n_samples

        # append segmented recordings and indices
        X_s.append(X_recording)
        y_s.append(np.tile(y[i], (n_seg, 1)))

    X_s = np.vstack(X_s)
    y_s = np.vstack(y_s)

    return X_s, y_s, names_s

def create_cwt_images(X, y, name, jpg_dir, fs=1000):
    n_samples = X.shape[0]

    dt = 1 / fs
    dj = 1 / 10
    fb = WaveletTransformTorch(dt, dj, cuda=True)
    batch_size = 64
    batch_start = 0
    batch_end = batch_size

    while batch_end < n_samples:
        # get segment data
        data = X[batch_start:batch_end, :]

        # apply CWT to data, take abs of coeffs
        cfs = fb.cwt(data)
        cfs = abs(cfs)

        for idx, cf in enumerate(cfs):
            # save cf as image here
            save_cfs_as_jpg(cf, y[batch_start+idx], name[batch_start+idx], im_dir=jpg_dir)

        batch_start += batch_size
        batch_end += batch_size

    # apply CWT to remaining data
    data = X[batch_start:n_samples, :]
    cfs = fb.cwt(data)
    cfs = abs(cfs)

    for idx, cf in enumerate(cfs):
        # jpg image
        save_cfs_as_jpg(cf, y[batch_start+idx,:], name[batch_start+idx], im_dir=jpg_dir)


def save_cfs_as_jpg(cfs, y, fname, im_dir):
    # extract label for saving
    classes = ['normal', 'abnormal']
    label = classes[np.argmax(y)]

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
    fname = os.path.join(save_dir, fname)
    img.save(fname + '.jpg')


def plot_images(X, y, fs=1000):
    classes = ['present', 'unknown', 'absent']

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
