import mne
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from mne_connectivity import SpectralConnectivity

# Define a wavelet denoising function
def wavelet_denoise(data, wavelet='db4', threshold=0.2):
    import pywt
    coeffs = pywt.wavedec(data, wavelet)
    thresholded_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(thresholded_coeffs, wavelet)

# Define a dynamic baseline adjustment function
def dynamic_baseline_adjust(data, window_size=256):
    baseline = gaussian_filter1d(data, sigma=window_size)
    return data - baseline

# Load CSV files
data1 = pd.read_csv('recording.csv')

# Extract EEG channels (ignoring 'time' and 'AUX')
data1_eeg = data1[['TP9', 'AF7', 'AF8', 'TP10']].values

# Define sampling frequency (replace 256 with your actual sampling rate)
sfreq = 256  # Hz
n_channels = data1_eeg.shape[1]  # Number of EEG channels

# Create channel names and types
ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
ch_types = ['eeg'] * n_channels

# Create MNE Info object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create RawArray
raw1 = mne.io.RawArray(data1_eeg.T, info)  # Transpose to match MNE's (n_channels, n_samples)

# Apply bandpass filter for alpha band (8-12 Hz)
raw1.filter(8, 12, fir_design='firwin')

# Extract filtered data
data1_eeg = raw1.get_data()

# Apply wavelet denoising
for i in range(data1_eeg.shape[0]):
    data1_eeg[i] = wavelet_denoise(data1_eeg[i])

# Apply dynamic baseline adjustment
for i in range(data1_eeg.shape[0]):
    data1_eeg[i] = dynamic_baseline_adjust(data1_eeg[i])

# Ensure the data is 3D: (1, n_channels, n_times)
data = data1_eeg[np.newaxis, :, :]  # Add an extra dimension for epochs

# Check the shape of the data
print(f"Data shape before passing to SpectralConnectivity: {data.shape}")

# Define epochs, channels, and times coordinates
epochs = ['Epoch 1']  # One epoch in this case
channels = ch_names  # Channel names from the data
times = np.arange(data.shape[2])  # Time points from the data

# Initialize the SpectralConnectivity object
conn = SpectralConnectivity(
    data,
    method='pli',  # Use 'pli' for Phase Lag Index
    sfreq=sfreq,
    freqs=[8, 12],  # Define the frequency band for alpha band (8-12 Hz)
    n_nodes=n_channels,  # Number of channels (nodes)
    dims=['epochs', 'channels', 'times'],  # Explicitly define the dimensions
    coords=[epochs, channels, times],  # Explicitly define the coordinates
    verbose=True
)

# Compute the connectivity
con = conn.get_data()  # Get the connectivity matrix

# Output the computed PLI
print("PLI Computed:")
print(con)
