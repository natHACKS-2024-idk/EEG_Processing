import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt, welch, savgol_filter
import matplotlib.pyplot as plt

# Load EEG data from two separate CSV files
df1 = pd.read_csv('recording11.csv')  # Dataset 1
df2 = pd.read_csv('recording12.csv')  # Dataset 2

# Assume both files contain 'AF7' and 'AF8' signals
eeg_signal_AF7_1 = df1['AF7'].values
eeg_signal_AF7_2 = df2['AF7'].values
eeg_signal_AF8_1 = df1['AF8'].values
eeg_signal_AF8_2 = df2['AF8'].values

# Ensure all signals have the same length
min_length = min(len(eeg_signal_AF7_1), len(eeg_signal_AF7_2), len(eeg_signal_AF8_1), len(eeg_signal_AF8_2))
eeg_signal_AF7_1 = eeg_signal_AF7_1[:min_length]
eeg_signal_AF7_2 = eeg_signal_AF7_2[:min_length]
eeg_signal_AF8_1 = eeg_signal_AF8_1[:min_length]
eeg_signal_AF8_2 = eeg_signal_AF8_2[:min_length]

# Sampling rate (Hz)
sampling_rate = 1000  # Example: 1000 samples per second

# Bandpass filter to isolate the frequency range of interest (e.g., alpha waves: 8-13 Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Apply Savitzky-Golay filter for denoising
def apply_savitzky_filter(data, window_length, polyorder):
    return savgol_filter(data, window_length, polyorder)

# Compute PLI for a pair of signals over multiple epochs
def compute_epoch_based_pli(signal_1, signal_2, epoch_length, fs):
    epoch_samples = int(epoch_length * fs)
    num_epochs = len(signal_1) // epoch_samples
    pli_values = []

    for epoch_idx in range(num_epochs):
        start = epoch_idx * epoch_samples
        end = start + epoch_samples
        print(f"Processing epoch {epoch_idx + 1}/{num_epochs}...")

        # Extract epoch
        epoch_signal_1 = signal_1[start:end]
        epoch_signal_2 = signal_2[start:end]

        # Apply bandpass filter
        filtered_signal_1 = bandpass_filter(epoch_signal_1, 8, 13, fs)
        filtered_signal_2 = bandpass_filter(epoch_signal_2, 8, 13, fs)

        # Apply Savitzky-Golay filter for further smoothing
        smoothed_signal_1 = apply_savitzky_filter(filtered_signal_1, window_length=51, polyorder=3)
        smoothed_signal_2 = apply_savitzky_filter(filtered_signal_2, window_length=51, polyorder=3)

        # Compute the analytic signal
        analytic_signal_1 = hilbert(smoothed_signal_1)
        analytic_signal_2 = hilbert(smoothed_signal_2)

        # Extract the instantaneous phases
        phase_1 = np.angle(analytic_signal_1)
        phase_2 = np.angle(analytic_signal_2)

        # Compute phase difference and PLI
        phase_diff = phase_1 - phase_2
        pli = np.abs(np.mean(np.sign(phase_diff)))
        pli_values.append(pli)

    print("PLI computation complete.")
    return np.mean(pli_values)

# Compute Welch PSD
def compute_psd(signal, fs, nperseg=1000):
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, window='hann')  # Use 'hann' window
    return freqs, psd

# Compute PLI and PSD for AF7
pli_AF7 = compute_epoch_based_pli(eeg_signal_AF7_1, eeg_signal_AF7_2, epoch_length=2, fs=sampling_rate)
freqs_AF7, psd_AF7_1 = compute_psd(eeg_signal_AF7_1, sampling_rate)
freqs_AF7, psd_AF7_2 = compute_psd(eeg_signal_AF7_2, sampling_rate)

# Compute PLI and PSD for AF8
pli_AF8 = compute_epoch_based_pli(eeg_signal_AF8_1, eeg_signal_AF8_2, epoch_length=2, fs=sampling_rate)
freqs_AF8, psd_AF8_1 = compute_psd(eeg_signal_AF8_1, sampling_rate)
freqs_AF8, psd_AF8_2 = compute_psd(eeg_signal_AF8_2, sampling_rate)

# Print results
print(f"Average PLI for AF7: {pli_AF7:.4f}")
print(f"Average PLI for AF8: {pli_AF8:.4f}")

# Plot PSD
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogy(freqs_AF7, psd_AF7_1, label='AF7 - Dataset 1')
plt.semilogy(freqs_AF7, psd_AF7_2, label='AF7 - Dataset 2')
plt.title('Power Spectral Density (AF7)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogy(freqs_AF8, psd_AF8_1, label='AF8 - Dataset 1')
plt.semilogy(freqs_AF8, psd_AF8_2, label='AF8 - Dataset 2')
plt.title('Power Spectral Density (AF8)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.legend()

plt.tight_layout()
plt.show()
