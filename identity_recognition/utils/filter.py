from scipy import signal
import numpy as np

def filter(data, low_freq, high_freq, sample_rate):
        b, a = signal.butter(4, 
                             [low_freq, high_freq], 
                             btype='bandpass', 
                             fs=sample_rate)
        filtered_signal = signal.lfilter(b, a, data, axis=0)

        return filtered_signal

def filter_multi(data, low_freq, high_freq, sample_rate):
    num_channel = data.shape[0]
    filtered_signal = np.zeros_like(data)
    for i in range(num_channel):
        filtered_signal[i] = filter(data[i], low_freq, high_freq, sample_rate)
    return filtered_signal