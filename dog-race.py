import time
import serial
import numpy as np

from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet, local_clock
from scipy.signal import firwin, lfilter, lfilter_zi, welch
from enum import Enum
from scipy.signal import butter, filtfilt

import sys

class State(Enum):
    BACKWARD = 1
    STAY = 2
    FORWARD = 3

any_key_pressed = False

ser = serial.Serial('/dev/rfcomm0', 115200)


# === Compute band power ===
def compute_band_power(epoch, band, fs, order=4):
    epoch = np.atleast_2d(epoch)
    if epoch.shape[0] < epoch.shape[1]:
        epoch = epoch.T  # Ensure shape is (samples, channels)
    
    nyq = 0.5 * fs
    low, high = band[0] / nyq, band[1] / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, epoch, axis=0)

    power = np.mean(filtered ** 2, axis=0)  # mean power per channel
    return power


# === Parameters ===
fs = 256
n_channels = 4
channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
band_defs = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
saturation_thresh = 50

bf = firwin(64, np.array([1, 40]) / (fs / 2.), width=0.05, pass_zero=False)
af = [1.0]
zi_init = lfilter_zi(bf, af)

print("🔍 Searching for EEG stream...")
eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
eeg_inlet = StreamInlet(eeg_streams[0])
marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string')
marker_outlet = StreamOutlet(marker_info)

data_f = np.zeros((int(fs * 4.0), n_channels))
filt_state = np.tile(zi_init, (n_channels, 1)).T
stds = [0] * n_channels
statuses = ["Unknown"] * n_channels

current_state = None
next_state = None

alpha_summary = None
feedback_buffer = []
last_feedback_time = time.time()
alpha_window = []
current_alpha = None

alpha_threshold = float(sys.argv[1])

#def state_change(alpha_power, band_lower=60, band_upper=80):
def state_change(alpha_power, threshold):
    if alpha_power <= threshold:
        return State.BACKWARD
    elif alpha_power >= threshold:
        return State.FORWARD
    else:
        return State.STAY

while True:
    samples, _ = eeg_inlet.pull_chunk(timeout=0.0, max_samples=fs)
    if samples:
        samples = np.array(samples)
        filt_chunk, filt_state = lfilter(bf, af, samples[:, :n_channels], axis=0, zi=filt_state)
        data_f = np.vstack([data_f, filt_chunk])[-int(fs * 4.0):]

        feedback_buffer.extend(filt_chunk.tolist())
        if len(feedback_buffer) > fs * 2:
            feedback_buffer = feedback_buffer[-fs * 2:]
        if time.time() - last_feedback_time >= 1.0 and len(feedback_buffer) >= fs * 2:
            feedback_array = np.array(feedback_buffer)
            all_channels = feedback_array[:, :]
            if np.std(all_channels) > 5:
                alpha_power = compute_band_power(all_channels, band_defs["Alpha"], fs).mean()

                alpha = compute_band_power(all_channels, band_defs["Alpha"], fs)
                theta = compute_band_power(all_channels, band_defs["Theta"], fs)
                beta  = compute_band_power(all_channels, band_defs["Beta"], fs)
                ratio = alpha / (theta * beta)
                alpha_window.append(ratio.mean())


#                alpha_window.append(alpha_power)
                if len(alpha_window) > 5:
                    alpha_window.pop(0)
                current_alpha = np.mean(alpha_window)

                next_state = state_change(current_alpha, alpha_threshold)
                if current_state != next_state:
                    current_state = next_state
                    print(f'Writing to serial {current_state}, {current_alpha} \r')
                    if current_state == State.BACKWARD:
                        ser.write("kbk".encode('utf-8'))
                    elif current_state == State.STAY:
                        ser.write('kbalance'.encode('utf-8'))
                    else:
                        ser.write('kwkF'.encode('utf-8'))
                    last_feedback_time = time.time()

        stds = np.std(data_f, axis=0, ddof=1)
        means = np.mean(data_f, axis=0)
        statuses = [
            "Disconnected" if std == 0 else
            "Noisy" if std > 30 else
            "Saturated" if abs(mean) >= saturation_thresh else
            "OK"
            for std, mean in zip(stds, means)
        ]


#for i in range(10):
#    u = np.random.uniform()
#    if u < 0.4:
#        ser.write("kbk".encode('utf-8'))
#        time.sleep(2)
#    elif u > 0.6:
#        ser.write('kwkF'.encode('utf-8'))
#        time.sleep(2)
#    else:
#        ser.write('kbalance'.encode('utf-8'))
#        time.sleep(2)
#


ser.write("kbalance".encode('utf-8'))
ser.close()
