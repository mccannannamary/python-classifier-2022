from helper_code import *
import numpy as np
from scipy import signal
from matplotlib import cm
from PIL import Image
from wavelets_pytorch.transform import WaveletTransform
from sklearn.model_selection import train_test_split
import glob, shutil
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from skorch.callbacks import Checkpoint
from pretrain_model_utils import ResNet18


# Apply preprocessing steps to signal
def preprocess(sig, fs_resample, fs):
    # resample signal
    sig = signal.resample_poly(sig, up=fs_resample, down=fs)

    # filter signal
    sig = butterworth_low_pass_filter(sig, 2, 400, fs_resample)
    sig = butterworth_high_pass_filter(sig, 2, 25, fs_resample)

    # remove spikes from signal
    sig = schmidt_spike_removal(sig, fs_resample)

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

    murmur_classes = ['Present', 'Unknown', 'Absent']
    n_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    n_outcome_classes = len(outcome_classes)

    # Extract the features and murmurs.
    if verbose >= 1:
        print('Extracting features and murmurs from the Challenge data...')

    recordings = list()
    features = list()
    murmurs = list()
    relabeled_murmurs = list()
    outcomes = list()
    rec_names = list()
    rec_names_list = list()

    for i in range(n_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # get current recording names
        current_recording_names = load_recording_names(current_patient_data)
        rec_names.append(np.vstack(current_recording_names))
        rec_names_list.append(current_recording_names)

        # Extract features
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # append processed recording from each location for this patient
        n_recordings = len(current_recordings)
        for r in range(n_recordings):
            recording = preprocess(current_recordings[r], fs_resample=fs_resample, fs=fs)
            recordings.append(recording)

        # Extract murmur for each recording using one-hot encoding
        current_murmurs = np.zeros(n_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmurs[j] = 1
        murmurs.append(np.tile(current_murmurs, (n_recordings, 1)))

        # Extract outcome for each recording using one-hot encoding
        current_outcome = np.zeros(n_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(np.tile(current_outcome, (n_recordings, 1)))

        if compare_strings(murmur, 'Present'):
            murmur_locations = get_murmur_locations(current_patient_data)
            if len(murmur_locations) != len(current_recordings):
                # default 'Absent' murmur
                current_murmurs = np.tile(np.array([0, 0, 1]), (n_recordings, 1))
                # correct murmurs so that only those where murmur was heard are actually labeled as positive
                for location in murmur_locations:
                    idx = [k for k, s in enumerate(rec_names_list[i]) if location in s]
                    current_murmurs[idx, :] = np.array([1, 0, 0])
                relabeled_murmurs.append(current_murmurs)

            else:
                # Extract murmur for each recording using one-hot encoding
                current_murmurs = np.zeros(n_murmur_classes, dtype=int)
                murmur = get_murmur(current_patient_data)
                if murmur in murmur_classes:
                    j = murmur_classes.index(murmur)
                    current_murmurs[j] = 1
                relabeled_murmurs.append(np.tile(current_murmurs, (n_recordings, 1)))

        else:
            # Extract murmur for each recording using one-hot encoding
            current_murmurs = np.zeros(n_murmur_classes, dtype=int)
            murmur = get_murmur(current_patient_data)
            if murmur in murmur_classes:
                j = murmur_classes.index(murmur)
                current_murmurs[j] = 1
            relabeled_murmurs.append(np.tile(current_murmurs, (n_recordings, 1)))

    features = np.vstack(features)
    murmurs = np.vstack(murmurs)
    relabeled_murmurs = np.vstack(relabeled_murmurs)
    outcomes = np.vstack(outcomes)
    rec_names = np.vstack(rec_names)

    return recordings, features, murmurs, relabeled_murmurs, outcomes, rec_names


def segment_challenge_data(X, y_murmurs, y_relabeled_murmurs, y_outcomes, rec_names):
    fs = 1000
    seg_len = 7.5

    n_recordings = len(X)
    n_samples = int(fs*seg_len)

    X_seg = list()
    y_murmurs_seg = list()
    y_murmurs_seg_relabel = list()
    y_outcomes_seg = list()
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
            X_recording[seg, :] = tmp
            # check if should be taking current_name[0], check concatenation
            names_seg.append(current_name[0] + '_' + str(seg).zfill(3))

            start_idx += n_samples
            end_idx += n_samples

        # recording too short, pad with zeros
        if n_segs == 0:
            X_recording = np.ndarray((1, n_samples))
            tmp = X[i]
            tmp = (tmp - np.mean(tmp)) / np.std(tmp)
            N = n_samples - len(tmp)
            X_recording[0, :] = np.concatenate([tmp, np.zeros(N)])
            names_seg.append(current_name[0] + '_' + str(0).zfill(3))
            n_segs += 1

        # append segmented recordings and labels
        X_seg.append(X_recording)
        y_murmurs_seg.append(np.tile(y_murmurs[i], (n_segs, 1)))
        y_murmurs_seg_relabel.append(np.tile(y_relabeled_murmurs[i], (n_segs, 1)))
        y_outcomes_seg.append(np.tile(y_outcomes[i], (n_segs, 1)))

    X_seg = np.vstack(X_seg)
    y_murmurs_seg = np.vstack(y_murmurs_seg)
    y_murmurs_seg_relabel = np.vstack(y_murmurs_seg_relabel)
    y_outcomes_seg = np.vstack(y_outcomes_seg)

    return X_seg, y_murmurs_seg, y_murmurs_seg_relabel, y_outcomes_seg, names_seg


def create_cwt_images(X, y_murmurs, y_relabeled_murmurs, y_outcomes, name, murmur_im_dir, murmur_im_dir_relabel, outcome_im_dir):
    fs = 1000

    n_samples = X.shape[0]

    dt = 1 / fs
    dj = 1 / 10
    fb = WaveletTransform(dt, dj)
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
            save_cfs_as_jpg(cf, y_murmurs[batch_start + idx, :],
                            y_relabeled_murmurs[batch_start + idx, :],
                            y_outcomes[batch_start + idx, :],
                            name[batch_start+idx],
                            murmur_im_dir=murmur_im_dir,
                            murmur_im_dir_relabel=murmur_im_dir_relabel,
                            outcome_im_dir=outcome_im_dir)

        batch_start += batch_size
        batch_end += batch_size

    # apply CWT to remaining data
    data = X[batch_start:n_samples, :]
    cfs = fb.cwt(data)
    cfs = abs(cfs)

    for idx, cf in enumerate(cfs):
        # jpg image
        save_cfs_as_jpg(cf, y_murmurs[batch_start + idx, :],
                        y_relabeled_murmurs[batch_start + idx, :],
                        y_outcomes[batch_start + idx, :],
                        name[batch_start+idx],
                        murmur_im_dir=murmur_im_dir,
                        murmur_im_dir_relabel=murmur_im_dir_relabel,
                        outcome_im_dir=outcome_im_dir)


def save_cfs_as_jpg(cfs, y_murmur, y_relabeled_murmur, y_outcome, fname, murmur_im_dir, murmur_im_dir_relabel, outcome_im_dir):
    # extract murmur_label for saving
    murmur_classes = ['present', 'unknown', 'absent']
    outcome_classes = ['abnormal', 'normal']
    murmur_label = murmur_classes[np.argmax(y_murmur)]
    murmur_relabel = murmur_classes[np.argmax(y_relabeled_murmur)]
    outcome_label = outcome_classes[np.argmax(y_outcome)]

    # rescale cfs to interval [0, 1]
    cfs = (cfs - cfs.min()) / (cfs.max() - cfs.min())

    # create colormap
    cmap = cm.get_cmap('binary')

    # apply colormap to data, return as ints from 0 to 255
    img = cmap(cfs, bytes=True)

    # convert from rgba to rgb
    img = np.delete(img, 3, 2)

    # create image from numpy array
    img = Image.fromarray(img)

    # resize the image
    img = img.resize((224, 224))

    # save image in appropriate class directory for murmurs
    save_dir = os.path.join(murmur_im_dir, murmur_label)
    os.makedirs(save_dir, exist_ok=True)
    tmp_fname = os.path.join(save_dir, fname)
    img.save(tmp_fname + '.jpg')

    # save image in appropriate class directory for relabeled murmurs
    save_dir = os.path.join(murmur_im_dir_relabel, murmur_relabel)
    os.makedirs(save_dir, exist_ok=True)
    tmp_fname = os.path.join(save_dir, fname)
    img.save(tmp_fname + '.jpg')

    # save image in appropriate class directory for outcomes
    save_dir = os.path.join(outcome_im_dir, outcome_label)
    os.makedirs(save_dir, exist_ok=True)
    tmp_fname = os.path.join(save_dir, fname)
    img.save(tmp_fname + '.jpg')


def split_data(data_folder, train_folder, val_folder, test_folder):

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

    ids_val, ids_test, labels_val, labels_test = \
        train_test_split(ids_val,
                         labels_val,
                         test_size=0.5,
                         random_state=1,
                         stratify=labels_val)

    # get all files matching ids_train and move to train folder (glob and shutils)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for pt_id in ids_train:
        tmp = data_folder + pt_id + '*'
        for file in glob.glob(tmp):
            shutil.copy(file, train_folder)

    for pt_id in ids_val:
        tmp = data_folder + pt_id + '*'
        for file in glob.glob(tmp):
            shutil.copy(file, val_folder)

    for pt_id in ids_test:
        tmp = data_folder + pt_id + '*'
        for file in glob.glob(tmp):
            shutil.copy(file, val_folder)


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


def get_murmur_locations(data):
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            murmur_locations = l.split(' ')[2]
            murmur_locations = murmur_locations.split('+')

    return murmur_locations


def train_net(train_set, valid_set, class_weights, scratch_name):

    # Create a torch.device() which should be the GPU if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaded_resnet18 = ResNet18(n_classes=2, pretrained_weights=False)
    loaded_resnet18.load_state_dict(torch.load('pretrained_resnet18'))

    scratch_folder = './' + scratch_name + '_scratch/'
    class_weights = torch.FloatTensor(class_weights)
    n_classes = len(class_weights)

    net = NeuralNetClassifier(
        module=loaded_resnet18,
        criterion=nn.CrossEntropyLoss(weight=class_weights),
        lr=0.001,
        batch_size=4,
        max_epochs=20,
        optimizer=optim.SGD,
        optimizer__momentum=0.9,
        optimizer__weight_decay=0.0005,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        iterator_train__num_workers=8,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=8,
        callbacks=[
            ('lr_scheduler',
             LRScheduler(policy='ReduceLROnPlateau', patience=3, factor=0.1)),
            ('checkpoint',
             Checkpoint(dirname=scratch_folder,
                        monitor='valid_acc_best',
                        f_params='model.pkl',
                        f_optimizer='opt.pkl',
                        f_criterion='criterion.pkl',
                        f_history='history.json')),
        ],
        device=device
    )

    # initialize neural network
    net.initialize()

    # load parameters from pretrained model
    pretrained_model_folder = './pretrain_resnet_unfreeze/'
    param_fname = os.path.join(pretrained_model_folder, 'model.pkl')
    net.load_params(f_params=param_fname)

    # change number of murmur_classes in classification layer
    n_in_features = net.module.model.fc.in_features
    net.module.model.fc = nn.Linear(in_features=n_in_features, out_features=n_classes)

    net.fit(train_set, y=None)

    return net
