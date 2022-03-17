import pywt

from helper_code import *
from team_code import *
import numpy as np
from scipy import signal
from matplotlib import cm
from PIL import Image
from wavelets_pytorch.transform import WaveletTransformTorch
from sklearn.model_selection import train_test_split
import glob, shutil

# Apply preprocessing steps to signal
def preprocess(sig, fs_resample, fs):
    # resample signal
    sig = signal.resample_poly(sig, up=fs_resample, down=fs)

    # filter signal
    sig = butterworth_low_pass_filter(sig, 2, 400, fs_resample)
    sig = butterworth_high_pass_filter(sig, 2, 25, fs_resample)

    # remove spikes from signal
    sig = schmidt_spike_removal(sig, fs_resample)

    # # try wavelet denoising
    # w = pywt.Wavelet('db14')
    # maxlev = pywt.dwt_max_level(len(sig), w.dec_len)

    return sig


# Low-pass filter signal at specified cutoff frequency
def butterworth_low_pass_filter(sig,order,fc,fs):
    [b, a] = signal.butter(order, fc, btype='lowpass', output='ba', fs=fs)
    lp_filtered_sig = signal.filtfilt(b, a, sig)

    return lp_filtered_sig


# High-pass filter signal at specified cutoff frequency
def butterworth_high_pass_filter(sig,order,fc,fs):
    [b, a] = signal.butter(order, fc, btype='highpass', output='ba', fs=fs)
    hp_filtered_sig = signal.filtfilt(b, a, sig)

    return hp_filtered_sig


# Remove spikes from signal
def schmidt_spike_removal(sig,fs):
    """Remove spikes in PCG signal as in
    Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J.
    (2010). Segmentation of heart sound recordings by a duration-dependent
    hidden Markov model. Physiological Measurement, 31(4), 513-29

    1. Divide recording into 500 ms windows
    2. Find max absolute amplitude in each window
    3. If at least one MAA exceeds three times the median MAA:
        a. Go to window with greatest MAA
        b. Isolate noise spike in that window
        c. Replace noise spike with zeros
        d. Go-to step 2
    4. Finished

    This code is adapted from schmidt_spike_removal.m by David Springer
    """

    win_len_samples = int(0.5*fs)

    if np.mod(len(sig), win_len_samples) != 0:
        sig = sig[0:-np.mod(len(sig), win_len_samples)]

    # arrange signal into frames each having win_len_samples,
    windows = sig.reshape((-1, win_len_samples))

    # get max abs val in each window (row)
    maa = np.max(np.abs(windows), axis=1)

    if maa.size != 0:
        # while there are still maa's greater than 3x median, remove spikes
        spikes = np.nonzero(maa > 3*np.median(maa))
        while spikes[0].size != 0:
            # get index of window with largest maa
            win_idx = maa.argmax()

            # get index of spike within window
            spike_idx = np.argmax(np.abs(windows[win_idx]))

            # get indices of zero crossings in this window
            zero_crossings = np.where(np.diff(np.sign(windows[win_idx])))
            zero_crossings = zero_crossings[0]

            # find zero crossings before and after spike
            first_idx = zero_crossings[zero_crossings < spike_idx]
            isempty = first_idx.size == 0
            if isempty:
                first_idx = 0
            else:
                first_idx = first_idx[-1]
            last_idx = zero_crossings[zero_crossings > spike_idx]
            isempty = last_idx.size == 0
            if isempty:
                last_idx = win_len_samples-1
            else:
                last_idx = last_idx[0]

            # set values in spike window to zero
            windows[win_idx][first_idx:last_idx+1] = 0

            # recalculate max abs val in each window (row)
            maa = np.max(np.abs(windows), axis=1)

            # recalculate spikes
            spikes = np.nonzero(maa > 3 * np.median(maa))

    sig = windows.reshape(windows.shape[0]*windows.shape[1])

    return sig


def load_recording_names(data):
    n_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:n_locations + 1]

    recording_names = list()
    for i in range(n_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        recording_name = recording_file.split('.')[0]
        recording_names.append(recording_name)

    return recording_names


def get_challenge_data(data_folder, verbose, fs_resample=1000, fs=4000):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    n_patient_files = len(patient_files)

    if n_patient_files == 0:
        raise Exception('No data was provided.')

    classes = ['Present', 'Unknown', 'Absent']
    n_classes = len(classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    recordings = list()
    features = list()
    labels = list()
    rec_names = list()

    for i in range(n_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # get current recording names
        current_recording_names = load_recording_names(current_patient_data)
        rec_names.append(np.vstack(current_recording_names))

        # Extract features
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # append processed recording from each location for this patient
        n_recordings = len(current_recordings)
        for r in range(n_recordings):
            recording = preprocess(current_recordings[r], fs_resample=fs_resample, fs=fs)
            recordings.append(recording)

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
    features = np.vstack(features)

    return recordings, features, labels, rec_names

def segment_challenge_data(X, y, rec_names):
    fs = 1000
    seg_len = 7.5

    n_recordings = len(X)
    n_samples = int(fs*seg_len)

    X_seg = list()
    y_seg = list()
    names_seg = list()

    for i in range(n_recordings):
        # load current patient data
        current_recording = X[i]
        current_name = rec_names[i]

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
            names_seg.append(current_name[0] + '_' + str(seg).zfill(3))

            start_idx += n_samples
            end_idx += n_samples

        # append segmented recordings and labels
        X_seg.append(X_recording)
        y_seg.append(np.tile(y[i], (n_segs, 1)))

    X_seg = np.vstack(X_seg)
    y_seg = np.vstack(y_seg)

    return X_seg, y_seg, names_seg


def create_cwt_images(X, y, name, jpg_dir):
    fs = 1000

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
            save_cfs_as_jpg(cf, y[batch_start + idx], name[batch_start + idx], im_dir=jpg_dir)

        batch_start += batch_size
        batch_end += batch_size

    # apply CWT to remaining data
    data = X[batch_start:n_samples, :]
    cfs = fb.cwt(data)
    cfs = abs(cfs)

    for idx, cf in enumerate(cfs):
        # jpg image
        save_cfs_as_jpg(cf, y[batch_start + idx, :], name[batch_start + idx], im_dir=jpg_dir)


def save_cfs_as_jpg(cfs, y, fname, im_dir):
    # extract label for saving
    classes = ['present', 'unknown', 'absent']
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

def split_data(data_folder, train_folder, val_folder):

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
    for pt_id in ids_train:
        os.makedirs(train_folder, exist_ok=True)
        tmp = data_folder + pt_id + '*'
        for file in glob.glob(tmp):
            shutil.copy(file, train_folder)

    for pt_id in ids_val:
        os.makedirs(val_folder, exist_ok=True)
        tmp = data_folder + pt_id + '*'
        for file in glob.glob(tmp):
            shutil.copy(file, train_folder)
