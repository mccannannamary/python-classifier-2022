import pywt

from helper_code import *
import numpy as np
from scipy import signal
import hsmm_utils
from matplotlib import cm
from PIL import Image
from wavelets_pytorch.transform import WaveletTransformTorch
from sklearn.utils.random import sample_without_replacement


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


# Extract features from the data.
def get_nn_features(data, recordings):
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

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant]))

    return np.asarray(features, dtype=np.float32)


def load_recording_names(data_folder, data):
    n_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:n_locations + 1]

    recording_names = list()
    for i in range(n_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        recording_name = recording_file.split('.')[0]
        recording_names.append(recording_name)

    return recording_names


def get_challenge_data(data_folder, verbose, fs_resample=1000, fs=2000):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    n_patient_files = len(patient_files)

    if n_patient_files == 0:
        raise Exception('No data was provided.')

    # preliminary, keep only small percent of data to speed up processing
    keep_percent = 0.1
    n_samples = int(keep_percent*n_patient_files)
    keep_patient_files = sample_without_replacement(n_patient_files, n_samples, random_state=1)

    classes = ['Present', 'Unknown', 'Absent']
    n_classes = len(classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    recordings = list()
    features = list()
    labels = list()
    rec_names = list()
    pt_ids = list()

    for _, i in enumerate(keep_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # get current recording names
        current_recording_names = load_recording_names(data_folder, current_patient_data)
        rec_names.append(np.vstack(current_recording_names))

        # get current patient id
        pt_id = get_patient_id(current_patient_data)
        pt_ids.append(pt_id)

        # Extract clinical features
        current_features = get_nn_features(current_patient_data, current_recordings)
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

    return recordings, features, labels, rec_names, pt_ids

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

def segment_fhs(X, y, rec_names):
    fs = 1000
    seg_len = 1.0

    n_pretrain_files = len(X)
    n_samples = int(fs*seg_len)

    X_fhs = list()
    y_fhs = list()
    names_fhs = list()

    for i in range(n_pretrain_files):
        # load current patient data
        current_recording = X[i]
        current_name = rec_names[i]

        lpf_recording = butterworth_low_pass_filter(current_recording, 5, 150, fs)
        assigned_states = hsmm_utils.segment(lpf_recording, fs=fs)

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

            names_fhs.append(current_name[0] + '_' + str(row).zfill(3))

        # append segmented recordings and labels
        X_fhs.append(X_recording)
        y_fhs.append(np.tile(y[i], (n_fhs, 1)))

    X_fhs = np.vstack(X_fhs)
    y_fhs = np.vstack(y_fhs)

    return X_fhs, y_fhs, names_fhs

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


def wden(x, tptr, sorh, scal, n, wname):
    # epsilon stands for a very small number
    eps = 2.220446049250313e-16

    # decompose the input signal. Symetric padding is given as a default.
    coeffs = pywt.wavedec(x, wname, 'sym', n)

    # threshold rescaling coefficients
    if scal == 'one':
        s = 1
    elif scal == 'sln':
        s = wnoisest(coeffs)
    elif scal == 'mln':
        s = wnoisest(coeffs, level=n)
    else:
        raise ValueError('Invalid value for scale, scal = %s' % (scal))

    # wavelet coefficients thresholding
    coeffsd = [coeffs[0]]
    for i in range(0, n):
        if tptr == 'sqtwolog' or tptr == 'minimaxi':
            th = thselect(x, tptr)
        else:
            if len(s) == 1:
                if s < sqrt(eps) * max(coeffs[1 + i]):
                    th = 0
                else:
                    th = thselect(coeffs[1 + i] / s, tptr)
            else:
                if s[i] < sqrt(eps) * max(coeffs[1 + i]):
                    th = 0
                else:
                    th = thselect(coeffs[1 + i] / s[i], tptr)

        ### DEBUG
        #        print "threshold before rescaling:", th
        ###

        # rescaling
        if len(s) == 1:
            th = th * s
        else:
            th = th * s[i]

        #### DEBUG
        #        print "threshold:", th
        ####

        coeffsd.append(array(wthresh(coeffs[1 + i], sorh, th)))

    # wavelet reconstruction
    xdtemp = pywt.waverec(coeffsd, wname, 'sym')

    # get rid of the extended part for wavelet decomposition
    # extlen = abs(len(x) - len(xdtemp)) / 2
    # xd = xdtemp[extlen:len(x) + extlen]
    xd = xdtemp

    return xd


#################################
#
# thselect(x, tptr) returns threshold x-adapted value using selection rule defined by string tptr.
#
# tptr = 'rigrsure', adaptive threshold selection using principle of Stein's Unbiased Risk Estimate.
#        'heursure', heuristic variant of the first option.
#        'sqtwolog', threshold is sqrt(2*log(length(X))).
#        'minimaxi', minimax thresholding.

def thselect(x, tptr):
    x = array(x)  # in case that x is not an array, convert it into an array
    l = len(x)

    if tptr == 'rigrsure':
        sx2 = [sx * sx for sx in absolute(x)]
        sx2.sort()
        cumsumsx2 = cumsum(sx2)
        risks = []
        for i in range(0, l):
            risks.append((l - 2 * (i + 1) + (cumsumsx2[i] + (l - 1 - i) * sx2[i])) / l)
        mini = argmin(risks)
        th = sqrt(sx2[mini])
    if tptr == 'heursure':
        hth = sqrt(2 * log(l))

        # get the norm of x
        normsqr = dot(x, x)
        eta = 1.0 * (normsqr - l) / l
        crit = (log(l, 2) ** 1.5) / sqrt(l)

        ### DEBUG
        #        print "crit:", crit
        #        print "eta:", eta
        #        print "hth:", hth
        ###

        if eta < crit:
            th = hth
        else:
            sx2 = [sx * sx for sx in absolute(x)]
            sx2.sort()
            cumsumsx2 = cumsum(sx2)
            risks = []
            for i in range(0, l):
                risks.append((l - 2 * (i + 1) + (cumsumsx2[i] + (l - 1 - i) * sx2[i])) / l)
            mini = argmin(risks)

            ### DEBUG
            #            print "risk:", risks[mini]
            #            print "best:", mini
            #            print "risks[222]:", risks[222]
            ###

            rth = sqrt(sx2[mini])
            th = min(hth, rth)
    elif tptr == 'sqtwolog':
        th = sqrt(2 * log(l))
    elif tptr == 'minimaxi':
        if l < 32:
            th = 0
        else:
            th = 0.3936 + 0.1829 * log(l, 2)
    else:
        raise ValueError('Invalid value for threshold selection rule, tptr = %s' % (tptr))

    return th


#################################
#
# wthresh(x, sorh, t) returns the soft (sorh = 'soft') or hard (sorh = 'hard')
# thresholding of x, the given input vector. t is the threshold.
# sorh = 'hard', hard trehsholding
# sorh = 'soft', soft thresholding
#

def wthresh(x, sorh, t):
    if sorh == 'hard':
        y = [e * (abs(e) >= t) for e in x]
    elif sorh == 'soft':
        y = [((e < 0) * -1.0 + (e > 0)) * ((abs(e) - t) * (abs(e) >= t)) for e in x]
    else:
        raise ValueError('Invalid value for thresholding type, sorh = %s' % (sorh))

    return y


#################################
#
# wnoisest(coeffs, level = None) estimates the variance(s) of the given detail(s)
#
# coeffs = [CAn, CDn, CDn-1, ..., CD1], multi-level wavelet coefficients
# level, decomposition level. None is the default.
#

def wnoisest(coeffs, level=None):
    l = len(coeffs) - 1

    if level == None:
        sig = [abs(s) for s in coeffs[-1]]
        stdc = median(sig) / 0.6745
    else:
        stdc = []
        for i in range(0, l):
            sig = [abs(s) for s in coeffs[1 + i]]
            stdc.append(median(sig) / 0.6745)

    return stdc


#################################
#
# median(data) returns the median of data
#
# data, a list of numbers.
#

def median(data):
    temp = data[:]
    temp.sort()
    dataLen = len(data)
    if dataLen % 2 == 0:  # even number of data points
        med = (temp[dataLen // 2 - 1] + temp[dataLen // 2]) // 2.0
    else:
        med = temp[dataLen // 2]

    return med
