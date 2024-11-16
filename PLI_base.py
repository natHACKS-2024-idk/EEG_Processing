import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt

# Load EEG data from two separate CSV files
df1 = pd.read_csv('recording11.csv')  # Contains data for AF7 and AF8
df2 = pd.read_csv('recording12.csv')  # Contains data for AF7 and AF8

# Assume both files contain 'AF7' and 'AF8' signals in their respective columns
eeg_signal_1_af7 = df1['AF7'].values
eeg_signal_1_af8 = df1['AF8'].values
eeg_signal_2_af7 = df2['AF7'].values
eeg_signal_2_af8 = df2['AF8'].values

# Ensure both signals have the same length
min_length = min(len(eeg_signal_1_af7), len(eeg_signal_1_af8), len(eeg_signal_2_af7), len(eeg_signal_2_af8))
eeg_signal_1_af7 = eeg_signal_1_af7[:min_length]
eeg_signal_1_af8 = eeg_signal_1_af8[:min_length]
eeg_signal_2_af7 = eeg_signal_2_af7[:min_length]
eeg_signal_2_af8 = eeg_signal_2_af8[:min_length]

# Sampling rate (Hz)
sampling_rate = 1000  # Example: 1000 samples per second

# Bandpass filter to isolate the frequency range of interest (e.g., alpha waves: 8-13 Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Apply bandpass filter to both EEG signals (8-13 Hz for alpha waves)
filtered_signal_1_af7 = bandpass_filter(eeg_signal_1_af7, 8, 13, sampling_rate)
filtered_signal_1_af8 = bandpass_filter(eeg_signal_1_af8, 8, 13, sampling_rate)
filtered_signal_2_af7 = bandpass_filter(eeg_signal_2_af7, 8, 13, sampling_rate)
filtered_signal_2_af8 = bandpass_filter(eeg_signal_2_af8, 8, 13, sampling_rate)

# Compute the analytic signal using Hilbert transform (to get the phase)
analytic_signal_1_af7 = hilbert(filtered_signal_1_af7)
analytic_signal_1_af8 = hilbert(filtered_signal_1_af8)
analytic_signal_2_af7 = hilbert(filtered_signal_2_af7)
analytic_signal_2_af8 = hilbert(filtered_signal_2_af8)

# Extract the instantaneous phases
phase_1_af7 = np.angle(analytic_signal_1_af7)
phase_1_af8 = np.angle(analytic_signal_1_af8)
phase_2_af7 = np.angle(analytic_signal_2_af7)
phase_2_af8 = np.angle(analytic_signal_2_af8)

# Compute the phase difference
phase_diff_af7 = phase_1_af7 - phase_2_af7
phase_diff_af8 = phase_1_af8 - phase_2_af8

# Compute the Phase Lag Index (PLI)
PLI_af7 = np.abs(np.mean(np.sign(phase_diff_af7)))
PLI_af8 = np.abs(np.mean(np.sign(phase_diff_af8)))

# Print the PLI values for both pairs
print(f"Phase Lag Index (PLI) between AF7 signals: {PLI_af7:.4f}")
print(f"Phase Lag Index (PLI) between AF8 signals: {PLI_af8:.4f}")

# Plot the smoothed signals and their phase difference
plt.figure(figsize=(12, 6))

# Plot the smoothed EEG signals for AF7 comparison
plt.subplot(3, 1, 1)
plt.plot(df1['time'][:min_length], filtered_signal_1_af7, label='EEG Channel 1 AF7 (recording8)', color='b')
plt.plot(df2['time'][:min_length], filtered_signal_2_af7, label='EEG Channel 2 AF7 (recording10)', color='r')
plt.title('Smoothed AF7 EEG Signals (Alpha Band: 8-13 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plot the phase difference for AF7 comparison
plt.subplot(3, 1, 2)
plt.plot(df1['time'][:min_length], phase_diff_af7, label='Phase Difference (AF7)', color='g')
plt.title('Phase Difference between AF7 Signals')
plt.xlabel('Time (s)')
plt.ylabel('Phase Difference (radians)')

# Plot the sign of the phase difference (used for PLI) for AF7 comparison
plt.subplot(3, 1, 3)
plt.plot(df1['time'][:min_length], np.sign(phase_diff_af7), label='Sign of Phase Difference (AF7)', color='m')
plt.title('Sign of Phase Difference for AF7')
plt.xlabel('Time (s)')
plt.ylabel('Sign')

plt.tight_layout()
plt.show()

# Plot the smoothed EEG signals for AF8 comparison
plt.figure(figsize=(12, 6))

# Plot the smoothed EEG signals for AF8 comparison
plt.subplot(3, 1, 1)
plt.plot(df1['time'][:min_length], filtered_signal_1_af8, label='EEG Channel 1 AF8 (recording8)', color='b')
plt.plot(df2['time'][:min_length], filtered_signal_2_af8, label='EEG Channel 2 AF8 (recording10)', color='r')
plt.title('Smoothed AF8 EEG Signals (Alpha Band: 8-13 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plot the phase difference for AF8 comparison
plt.subplot(3, 1, 2)
plt.plot(df1['time'][:min_length], phase_diff_af8, label='Phase Difference (AF8)', color='g')
plt.title('Phase Difference between AF8 Signals')
plt.xlabel('Time (s)')
plt.ylabel('Phase Difference (radians)')

# Plot the sign of the phase difference (used for PLI) for AF8 comparison
plt.subplot(3, 1, 3)
plt.plot(df1['time'][:min_length], np.sign(phase_diff_af8), label='Sign of Phase Difference (AF8)', color='m')
plt.title('Sign of Phase Difference for AF8')
plt.xlabel('Time (s)')
plt.ylabel('Sign')

plt.tight_layout()
plt.show()
