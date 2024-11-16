import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure the CSV file exists
file = 'recording.csv'

if not os.path.exists(file):
    print(f"{file} not found.")
    exit()

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv(file)

# Verify that the required columns are present
if 'AF7' not in df.columns or 'AF8' not in df.columns:
    print(f"Required columns ('AF7' and 'AF8') not found in {file}.")
    exit()

# Step 2: Select and clean the data (remove NaN values, convert to numeric)
df = df.dropna(subset=['AF7', 'AF8'])  # Remove rows with NaN in critical columns

x = pd.to_numeric(df['AF7'], errors='coerce')  # Convert to numeric, coercing errors to NaN
y = pd.to_numeric(df['AF8'], errors='coerce')

# Remove NaN values after coercion
x = x.dropna()
y = y.dropna()

# Step 3: Downsample the data (optional)
# Downsample to every 10th sample (you can change this factor depending on your data)
downsample_factor = 10
x_downsampled = x[::downsample_factor]
y_downsampled = y[::downsample_factor]
time_downsampled = range(len(x_downsampled))  # Create a time axis for downsampled data

# Step 4: Create a wider figure for better clarity
plt.figure(figsize=(14, 10))  # Make the figure even taller for two plots (14 inches wide, 10 inches high)

# Step 5: Create the first plot for AF7
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
plt.plot(time_downsampled, x_downsampled, label=f"AF7 Waveform", color='blue', linestyle='-', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('AF7 Waveform')
plt.legend()
plt.grid(True)

# Step 6: Create the second plot for AF8
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
plt.plot(time_downsampled, y_downsampled, label=f"AF8 Waveform", color='red', linestyle='-', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('AF8 Waveform')
plt.legend()
plt.grid(True)

# Step 7: Adjust layout for better spacing between the plots
plt.tight_layout()

# Step 8: Display the plot
plt.show()
