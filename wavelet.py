import numpy as np
import pandas as pd
import pywt  # For wavelet denoising
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt

# Load EEG data from two separate CSV files
df1 = pd.read_csv('recording8.csv')  # Contains data for AF7
df2 = pd.read_csv('recording10.csv')  # Contains data for AF8

# Assume both files contain 'AF7' and 'AF8' signals in their respective columns
eeg_signal_1 = df1['AF7'].values
eeg_signal_2 = df2['AF8'].values

# Ensure both signals have the same length
min_length = min(len(eeg_signal_1), len(eeg_signal_2))
eeg_signal_1 = eeg_signal_1[:min_length]  # Truncate signal 1
eeg_signal_2 = eeg_signal_2[:min_length]  # Truncate signal 2

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
filtered_signal_1 = bandpass_filter(eeg_signal_1, 8, 13, sampling_rate)
filtered_signal_2 = bandpass_filter(eeg_signal_2, 8, 13, sampling_rate)

# Wavelet Denoising
def wavelet_denoise(signal, wavelet='db4', level=4, threshold=0.1):
    """
    Perform wavelet denoising on the signal.
    - wavelet: The wavelet type (default: 'db4')
    - level: Decomposition level (default: 4)
    - threshold: Threshold for detail coefficients (default: 0.1)
    """
    # Perform discrete wavelet transform (DWT)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Thresholding the detail coefficients
    coeffs_thresholded = [coeffs[0]]  # Approximation coefficients remain unchanged
    for i in range(1, len(coeffs)):
        # Apply soft thresholding to the detail coefficients
        coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    
    # Reconstruct the signal using the inverse DWT with thresholded coefficients
    cleaned_signal = pywt.waverec(coeffs_thresholded, wavelet)
    return cleaned_signal[:len(signal)]  # Ensure the signal length remains unchanged

# Apply Wavelet Denoising to both EEG signals
cleaned_signal_1 = wavelet_denoise(filtered_signal_1, wavelet='db4', level=4, threshold=0.1)
cleaned_signal_2 = wavelet_denoise(filtered_signal_2, wavelet='db4', level=4, threshold=0.1)

# Compute the analytic signal using Hilbert transform (to get the phase)
analytic_signal_1 = hilbert(cleaned_signal_1)
analytic_signal_2 = hilbert(cleaned_signal_2)

# Extract the instantaneous phases
phase_1 = np.angle(analytic_signal_1)
phase_2 = np.angle(analytic_signal_2)

# Compute the phase difference
phase_diff = phase_1 - phase_2

# Compute the Phase Lag Index (PLI)
PLI = np.abs(np.mean(np.sign(phase_diff)))

# Print the PLI value
print(f"Phase Lag Index (PLI) between the two signals: {PLI:.4f}")

# Plot the cleaned signals and their phase difference
plt.figure(figsize=(12, 6))

# Plot the cleaned EEG signals
plt.subplot(3, 1, 1)
plt.plot(df1['time'][:min_length], cleaned_signal_1, label='EEG Channel 1 (AF7)', color='b')
plt.plot(df2['time'][:min_length], cleaned_signal_2, label='EEG Channel 2 (AF8)', color='r')
plt.title('Cleaned EEG Signals (Alpha Band: 8-13 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plot the phase difference
plt.subplot(3, 1, 2)
plt.plot(df1['time'][:min_length], phase_diff, label='Phase Difference (AF7 - AF8)', color='g')
plt.title('Phase Difference between EEG Channels')
plt.xlabel('Time (s)')
plt.ylabel('Phase Difference (radians)')

# Plot the sign of the phase difference (used for PLI)
plt.subplot(3, 1, 3)
plt.plot(df1['time'][:min_length], np.sign(phase_diff), label='Sign of Phase Difference', color='m')
plt.title('Sign of Phase Difference')
plt.xlabel('Time (s)')
plt.ylabel('Sign')

plt.tight_layout()
plt.show()
