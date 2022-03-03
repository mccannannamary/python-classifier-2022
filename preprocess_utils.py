import numpy as np
from scipy import signal

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

    sig = sig[0:-np.mod(len(sig),win_len_samples)]

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