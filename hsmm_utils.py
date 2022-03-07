import sys

from scipy import signal
from scipy.stats import norm
import preprocess_utils
import numpy as np
import pywt
import matplotlib.pyplot as plt
import math
from numpy.linalg import det, inv

# parameter definitions
features_fs = 50
segmentation_tol = 0.1  # seconds

B = np.array([[0.6386, 0.0918, 0.2460, 0.2619],
              [-2.2125, 1.5887, -0.4313, 2.8046],
              [1.2804, -0.7538, -0.1327, -1.2676],
              [-0.0786, 0.5482, 0.2668, 0.0263],
              [-0.0500, 0.2607, -0.2469, 0.4411]])

pi = np.array([0.25, 0.25, 0.25, 0.25])

mu = np.array([3.0583e-15, 4.9309e-16, -6.2185e-18, -1.5315e-15])

Sigma = np.array([[0.9984, 0.9035, 0.7426, 0.7069],
                  [0.9035, 0.9984, 0.8504, 0.7941],
                  [0.7426, 0.8504, 0.9984, 0.7068],
                  [0.7069, 0.7941, 0.7068, 0.9984]])

realmin = np.finfo(np.double).tiny

def segment(sig, fs):
    features = get_springer_features(sig, fs)

    heart_rate, sys_time_interval = get_heart_rate_schmidt(sig, fs)

    qt = viterbi_decode_springer(features,
                                 heart_rate,
                                 sys_time_interval,
                                 features_fs)

    assigned_states = expand_qt(qt, features_fs, fs, len(sig))

    return assigned_states


def get_springer_features(sig, fs):
    # get homomorphic & hilbert envelopes
    homomorphic_env = homomorphic_envelope(sig, fs)
    hilbert_env = hilbert_envelope(sig)

    # downsample envelopes
    homomorphic_env = signal.decimate(homomorphic_env, int(fs / features_fs), ftype='fir')
    hilbert_env = signal.decimate(hilbert_env, int(fs / features_fs), ftype='fir')

    # normalize envelopes
    homomorphic_env = normalize_signal(homomorphic_env)
    hilbert_env = normalize_signal(hilbert_env)

    # power spectral density features
    psd = psd_features(sig, fs)
    psd = signal.resample_poly(psd, up=len(homomorphic_env), down=len(psd))
    psd = normalize_signal(psd)

    # wavelet feature
    wavelet = wavelet_features(sig)
    wavelet = signal.resample_poly(wavelet, up=features_fs, down=fs)
    wavelet = normalize_signal(wavelet)

    # fig, axs = plt.subplots(4)
    # axs[0].plot(homomorphic_env)
    # axs[0].set_title('Homomorphic Envelope')
    # axs[1].plot(hilbert_env)
    # axs[1].set_title('Hilbert Envelope')
    # axs[2].plot(psd)
    # axs[2].set_title('PSD')
    # axs[3].plot(wavelet)
    # axs[3].set_title('Wavelet')
    # plt.show()
    # plt.close()
    #
    features = np.column_stack((homomorphic_env, hilbert_env, psd, wavelet))

    return features


def homomorphic_envelope(sig, fs, lpf_freq=8):
    """
    Extract homomorphic envelope of a signal, as in:

    Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J.
    (2010). Segmentation of heart sound recordings by a duration-dependent
    hidden Markov model. Physiological Measurement, 31(4), 513-29

    Gupta, C. N., Palaniappan, R., Swaminathan, S., & Krishnan, S. M. (2007).
    Neural network classification of homomorphic segmented heart sounds.
    Applied soft computing, 7(1), 286-297.

    A zero-phase low-pass Butterworth filter is used to extract the envelope
    """
    # low pass filter at lpf_freq cutoff
    input_sig = np.log(np.abs(signal.hilbert(sig)))
    sig = preprocess_utils.butterworth_low_pass_filter(input_sig, 1, lpf_freq, fs)

    # homomorphic envelope
    env = np.exp(sig)

    # remove spurious spike in first sample
    env[0] = env[1]

    return env


def hilbert_envelope(sig):
    return np.abs(signal.hilbert(sig))


def normalize_signal(sig):
    # zero-mean signal
    sig = sig - np.mean(sig)
    # standardize to unit variance
    sig = sig / np.std(sig)

    return sig


def psd_features(sig, fs, freq_lim_low=40, freq_lim_high=60):
    """
    Extract PSD features for segmentation

    input data: heart-sound waveform
    input fs:
    input freq_lim_low: lower bound of analyzed freq range
    input freq_lim_high: upper bound of analyzed freq range

    output psd: array of mean PSD values between max and min
    limits in each stft time window
    """
    f, t, sxx = signal.spectrogram(sig, fs,
                                   window='hamming',
                                   nperseg=int(fs / 40),
                                   noverlap=round(fs / 80),
                                   nfft=1000,
                                   mode='psd')

    # get indices of frequency range of interest
    lim_low_idx = np.argmin(np.abs(f - freq_lim_low))
    lim_high_idx = np.argmin(np.abs(f - freq_lim_high))

    # extract PSDs in freq range of interest
    sxx = sxx[lim_low_idx:lim_high_idx, :]

    # get mean PSD value in each window
    psd = np.mean(sxx, axis=0)

    return psd


def wavelet_features(sig, wavelet='rbio3.9', level=3):
    # pad signal to be a multiple of 2**'wavelet_level'
    num_extra_samples = np.mod(len(sig), 2 ** level)
    sig = np.concatenate((sig, np.zeros(num_extra_samples)))

    ca = []
    cd = []
    coeffs = pywt.swt(sig, wavelet=wavelet, level=level)
    for a, d in reversed(coeffs):
        ca.append(a)
        cd.append(d)

    # extract detail coeffs at indicated level
    wavelet_features = np.abs(cd[level - 1])

    # remove padded samples
    wavelet_features = wavelet_features[:-4]

    return wavelet_features


def get_heart_rate_schmidt(sig, fs):
    """Estimate heart rate and systolic time interval from a PCG recording, used
    for the duration-dependent HSMM based segmentation of the PCG recording. The
    is based on the analysis of the autocorrelation function and positions of
    its peaks. Code is derived from Springer's Matlab implementation based on:

    Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J.
    (2010). Segmentation of heart sound recordings by a duration-dependent
    hidden Markov model. Physiological Measurement, 31(4), 513-29

    output heart_rate: HR of PCG in bpm
    output sys_time_interval: duration of systole in seconds, as derived from
    autocorrelation function.
    """

    # ========================================================================= #
    # Estimate HR from autocorr of signal envelope as peak in autocorr occuring
    # between 500-2000 ms after zero lag peak (Schmidt) (allowed HR = 30-120 bpm)
    # ========================================================================= #
    # get homomorphic env
    homomorphic_env = homomorphic_envelope(sig, fs)
    homomorphic_env = homomorphic_env - np.mean(homomorphic_env)

    # calculate its autocorr
    corr_hh = np.correlate(homomorphic_env, homomorphic_env, mode='full')
    # normalize so that autocorr at zero lag = 1
    corr_hh = corr_hh / np.max(corr_hh)
    # take only zero and positive lags
    corr_hh = corr_hh[np.argmax(corr_hh):]

    # find autocorr peak 500-2000 ms after zero lag
    min_idx = round(0.5 * fs)
    max_idx = 2 * fs

    first_hr_peak_idx = np.argmax(corr_hh[min_idx:max_idx + 1])
    first_hr_peak_idx = first_hr_peak_idx + min_idx

    heart_rate = 60 / (first_hr_peak_idx / fs)

    # ========================================================================= #
    # Find systolic time interval, defined as the time from lag zero to highest
    # autocorr peak between 200 ms and half the heart cycle (HR) duration
    # ========================================================================= #
    min_idx = round(0.2 * fs)
    max_idx = round(first_hr_peak_idx / 2)

    sys_peak = np.argmax(corr_hh[min_idx:max_idx + 1])
    sys_peak = sys_peak + min_idx

    sys_time_interval = sys_peak / fs

    return heart_rate, sys_time_interval


def viterbi_decode_springer(features,
                            heart_rate,
                            sys_time_interval,
                            fs):
    """Calculate the delta, psi, and qt matrices associated with Viterbi decoding
    algorithm from:
    Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.
    using eqs. 32a-35, and eqs 68-69 to include duration dependency of the states

    Decoding is performed after observation probabilities have been derived from
    the logistic regression model of:
    D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
    Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.

    And the function is extended to allow duration distributions to extend past the
    beginning and end of the sequence, allowing duration distributions to extend past
    the beginning and end, but only considering the observations within the sequence
    for emission probability estimation. Otherwise, the label sequence would need to
    start and stop with an "entire" state duration being fulfilled.

    input features: sequences of observed features (in columns)
    input pi: array of trained initial state probabilities
    input B: observation probabilities
    input obs_dis: observation distribution
    input heart_rate: estimate of HR
    input sys_time: estimate of systole duration
    input fs: sampling frequency of observation sequence

    output delta:
    output psi:
    output qt:

    Note there is no need for the A transition matrix, since transition between states
    is only dependent on state durations
    """

    # Some parameters
    obs_len = features.shape[0]
    n_states = 4

    # set max state duration to entire heart cycle
    max_state_dur = round(60 / heart_rate * fs)

    # initialize vars needed to find optimal state path along observation sequence
    delta = np.ones((obs_len + max_state_dur - 1, n_states)) * -math.inf
    psi = np.zeros((obs_len + max_state_dur - 1, n_states), dtype=int)
    psi_duration = np.zeros((obs_len + max_state_dur - 1, n_states), dtype=int)
    obs_probs = np.zeros((obs_len, n_states))

    # use Bayes law to get MAP estimate P(o|state) = P(state|obs) * P(obs) / P(states)
    # P(obs) is derived from MVN dist from all obs, and P(states) taken from pi
    for state in range(n_states):
        # add term for bias
        lg_features = np.column_stack((np.ones(features.shape[0]), features))

        # calculate lg probability
        pi_hat = 1 / (1 + np.exp(-lg_features @ B[:, state][:, None]))

        for n in range(obs_len):
            po_correction = mvn(features[n, :], mean=mu, cov=Sigma)

            obs_probs[n, state] = ((1 - pi_hat[n]) * po_correction) / pi[state]

    # Setting up state duration probabilities, using Gaussian dists
    d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole = \
        get_duration_distributions(heart_rate, sys_time_interval)

    duration_probs = np.zeros((n_states, 3 * features_fs))
    duration_sum = np.zeros((n_states, 1))

    for state in range(n_states):
        for d in range(max_state_dur):
            if state == 0:
                if d < min_S1 or d > max_S1:
                    duration_probs[state, d] = realmin
                else:
                    duration_probs[state, d] = norm.pdf(d, loc=d_distributions[state, 0],
                                                        scale=d_distributions[state, 1])

            elif state == 1:
                if d < min_systole or d > max_systole:
                    duration_probs[state, d] = realmin
                else:
                    duration_probs[state, d] = norm.pdf(d, loc=d_distributions[state, 0],
                                                        scale=d_distributions[state, 1])

            elif state == 2:
                if d < min_S2 or d > max_S2:
                    duration_probs[state, d] = realmin
                else:
                    duration_probs[state, d] = norm.pdf(d, loc=d_distributions[state, 0],
                                                        scale=d_distributions[state, 1])

            elif state == 3:
                if d < min_diastole or d > max_diastole:
                    duration_probs[state, d] = realmin
                else:
                    duration_probs[state, d] = norm.pdf(d, loc=d_distributions[state, 0],
                                                        scale=d_distributions[state, 1])

        # CHECK following 2 lines
        duration_sum[state] = np.sum(duration_probs[state, :])

    if len(duration_probs > 3 * fs):
        duration_probs[:, 3 * fs:] = []

    # Perform viterbi recursion (message passing)
    qt = np.zeros(len(delta))

    # eq. 32a and 69a, but leave out prob of being in state i for only 1 sample,
    # since state could have started before t=0. obs[0,:] is prob of initially
    # being in each state * prob of first obs coming from each state
    delta[0, :] = np.log(pi) + np.log(obs_probs[0, :])

    psi[0, :] = -1

    # change A to have zeros along diag, only duration probs and observation probs
    # influence change between states (only valid for states with distinct order)
    A = np.array([[0, 1, 0, 0],  # in S1 (state 0), go to systole (state 1)
                  [0, 0, 1, 0],  # in systole, go to S2
                  [0, 0, 0, 1],
                  [1, 0, 0, 0]])

    # precompute matrices used in loop below when possible, use realmin to avoid
    # divide by zero
    logA = np.log(A + realmin)
    duration_log = np.log((duration_probs / duration_sum) + realmin)

    for t in range(1, obs_len + max_state_dur - 1):
        for state in range(n_states):
            for dur in range(1, max_state_dur + 1):

                # start of analysis window, equals current time step minus horizon
                # we are looking max
                start_t = t - dur
                if start_t < 0:
                    start_t = 0
                elif start_t > obs_len - 1:
                    start_t = obs_len - 1

                # end of analysis window, equals current time step
                end_t = t
                if end_t > obs_len:
                    end_t = obs_len

                max_delta = np.max(delta[start_t, :] + logA[:, state])
                max_idx = np.argmax(delta[start_t, :] + logA[:, state])

                probs = np.prod(obs_probs[start_t:end_t, state])

                if probs == 0:
                    probs = realmin
                emission_probs = np.log(probs)

                if emission_probs == 0:
                    emission_probs = realmin

                # delta_temp = max_delta + emission_probs + np.log(duration_probs[state,dur]/duration_sum[state])
                delta_temp = max_delta + emission_probs + duration_log[state, dur]

                if delta_temp > delta[t, state]:
                    delta[t, state] = delta_temp
                    psi[t, state] = max_idx
                    psi_duration[t, state] = dur

    # find only delta after end of signal
    temp_delta = delta[obs_len:, :]

    state = np.argmax(np.max(temp_delta, axis=0))
    pos = np.argmax(np.max(temp_delta, axis=1))

    offset = pos + obs_len
    preceding_state = psi[offset, state]

    onset = offset - psi_duration[offset, state] + 1

    qt[onset:offset] = state

    state = preceding_state

    count = 0
    while onset > 1:
        offset = onset

        preceding_state = psi[offset, state]

        onset = offset - psi_duration[offset, state]

        if onset < 1:
            onset = 0

        qt[onset:offset] = state
        state = preceding_state
        count += 1

        if count > 1000:
            break

    qt = qt[0:obs_len]
    # plt.plot(qt)
    # plt.show()

    return qt


def get_duration_distributions(heart_rate, sys_time_interval):
    mean_S1 = round(0.122 * features_fs)
    std_S1 = round(0.022 * features_fs)
    mean_S2 = round(0.094 * features_fs)
    std_S2 = round(0.022 * features_fs)

    mean_systole = round(sys_time_interval * features_fs) - mean_S1
    std_systole = (25 / 1000) * features_fs
    mean_diastole = ((60 / heart_rate) - sys_time_interval - 0.094) * features_fs
    std_diastole = 0.07 * mean_diastole + (6 / 1000) * features_fs

    # Assign mean and covariance vals to d_distributions
    d_distributions = np.zeros((4, 2))
    d_distributions[0, 0] = mean_S1
    d_distributions[0, 1] = std_S1 ** 2

    d_distributions[1, 0] = mean_systole
    d_distributions[1, 1] = std_systole ** 2

    d_distributions[2, 0] = mean_S2
    d_distributions[2, 1] = std_S2 ** 2

    d_distributions[3, 0] = mean_diastole
    d_distributions[3, 1] = std_diastole ** 2

    # min systole and diastole times
    min_systole = mean_systole - 3 * (std_systole + std_S1)
    max_systole = mean_systole + 3 * (std_systole + std_S1)

    min_diastole = mean_diastole - 3 * std_diastole
    max_diastole = mean_diastole + 3 * std_diastole

    # set min and max for S1 and S2 sounds
    min_S1 = mean_S1 - 3 * std_S1
    if min_S1 < features_fs / 50:
        min_S1 = features_fs / 50

    min_S2 = mean_S2 - 3 * std_S2
    if min_S2 < features_fs / 50:
        min_S2 = features_fs / 50

    max_S1 = mean_S1 + 3 * std_S1
    max_S2 = mean_S2 + 3 * std_S2

    return d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole


def mvn(x, mean, cov):
    k = x.shape[-1]
    den = np.sqrt((2 * np.pi) ** k * det(cov))
    x = x - mean

    return np.squeeze(np.exp(-x[..., None, :] @ inv(cov) @ x[..., None] / 2)) / den


def expand_qt(qt, old_fs, new_fs, sig_len):
    """Upsample HSMM states to signal sampling frequency, so each sample is
    assigned a state

    input qt: HSMM assigned hidden states, from downsampled feature vector
    input old_fs: sampling freq of downsampled feature vector
    input new_fs: sampling freq of signal from which states derived
    input new_length: signal length

    output expanded_qt: upsampled state vector, one state per signal

    """
    original_qt = qt
    expanded_qt = np.zeros(sig_len)

    # find indices where state changes
    change_idx = np.nonzero(np.diff(original_qt))
    change_idx = change_idx[0]
    change_idx = np.concatenate((change_idx, np.array([len(original_qt)])))

    start_idx = 0
    count = 0
    for idx in change_idx:

        end_idx = idx

        mid_point = round((end_idx - start_idx) / 2) + start_idx
        mid_point_val = original_qt[mid_point]

        expanded_start_idx = round(start_idx / old_fs * new_fs)
        expanded_end_idx = round(end_idx / old_fs * new_fs)

        if expanded_end_idx > sig_len - 1:
            expanded_end_idx = sig_len

        expanded_qt[expanded_start_idx:expanded_end_idx] = mid_point_val

        start_idx = end_idx

    # plt.plot(expanded_qt)
    # plt.show()

    return expanded_qt

def get_states(assigned_states):

    assigned_states = assigned_states + 1

    idx = np.where(np.abs(np.diff(assigned_states)) > 0)[0]

    first_state = assigned_states[0]

    if first_state > 0:
        if first_state == 4:
            K = 0
        if first_state == 3:
            K = 1
        if first_state == 2:
            K = 2
        if first_state == 1:
            K = 3

    else:
        first_state = assigned_states[idx[0]+1]
        if first_state == 4:
            K = 1
        if first_state == 3:
            K = 2
        if first_state == 2:
            K = 3
        if first_state == 1:
            K = 0
        K = K+1

    indx2 = idx[K:]
    rem = np.mod(len(indx2), 4)
    if rem != 0:
        indx2 = indx2[:-rem]
    idx_states = np.reshape(indx2, (len(indx2) // 4, 4))
    return idx_states