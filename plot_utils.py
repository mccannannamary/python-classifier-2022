import numpy as np
import matplotlib.pyplot as plt

def plot_segmentations(current_recording, assigned_states):
    colors = ['C0', 'C1', 'C2', 'C3']

    s1_idx = np.nonzero(assigned_states == 0)[0]
    systole_idx = np.nonzero(assigned_states == 1)[0]
    s2_idx = np.nonzero(assigned_states == 2)[0]
    diastole_idx = np.nonzero(assigned_states == 3)[0]

    plt.plot(s1_idx,current_recording[s1_idx], color=colors[0], label='s1')
    plt.plot(systole_idx, current_recording[systole_idx], color=colors[1], label='sys')
    plt.plot(s2_idx, current_recording[s2_idx], color=colors[2], label='s2')
    plt.plot(diastole_idx, current_recording[diastole_idx], color=colors[3], label='dias')
    plt.legend()
    plt.show()
    plt.close()
