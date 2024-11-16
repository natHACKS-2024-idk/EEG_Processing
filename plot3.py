import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure both CSV files exist
file1 = 'recording.csv'
file2 = 'recording2.csv'

if not os.path.exists(file1):
    print(f"{file1} not found.")
    exit()

if not os.path.exists(file2):
    print(f"{file2} not found.")
    exit()

# Step 1: Load the two CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Verify columns in both files
for df, file_name in zip([df1, df2], [file1, file2]):
    if 'AF7' not in df.columns or 'AF8' not in df.columns:
        print(f"Required columns ('AF7' and 'AF8') not found in {file_name}.")
        exit()

# Step 2: Select and clean the data (remove NaN values, convert to numeric)
df1 = df1.dropna(subset=['AF7', 'AF8'])
df2 = df2.dropna(subset=['AF7', 'AF8'])

x1 = pd.to_numeric(df1['AF7'], errors='coerce')
y1 = pd.to_numeric(df1['AF8'], errors='coerce')

x2 = pd.to_numeric(df2['AF7'], errors='coerce')
y2 = pd.to_numeric(df2['AF8'], errors='coerce')

# Remove NaN values after coercion
x1 = x1.dropna()
y1 = y1.dropna()

x2 = x2.dropna()
y2 = y2.dropna()

# Step 3: Downsample the data to make the troughs wider
# Increase the downsampling factor to make the waves wider (less points, more spacing)
downsample_factor = 25  # Increased factor for wider troughs
x1_downsampled = x1[::downsample_factor]
y1_downsampled = y1[::downsample_factor]
x2_downsampled = x2[::downsample_factor]
y2_downsampled = y2[::downsample_factor]

time1_downsampled = range(len(x1_downsampled))  # Create a time axis for downsampled data
time2_downsampled = range(len(x2_downsampled))  # Create a time axis for downsampled data

# Step 4: Create a wider figure for better clarity
plt.figure(figsize=(14, 6))  # Make the figure wider for two plots (14 inches wide, 6 inches high)

# Step 5: Create the first plot for AF7 from both files
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot for AF7
plt.plot(time1_downsampled, x1_downsampled, label=f"AF7 from {file1}", color='blue', linestyle='-', linewidth=2)
plt.plot(time2_downsampled, x2_downsampled, label=f"AF7 from {file2}", color='green', linestyle='-', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('AF7 Comparison')
plt.legend()
plt.grid(True)

# Step 6: Create the second plot for AF8 from both files
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot for AF8
plt.plot(time1_downsampled, y1_downsampled, label=f"AF8 from {file1}", color='red', linestyle='-', linewidth=2)
plt.plot(time2_downsampled, y2_downsampled, label=f"AF8 from {file2}", color='purple', linestyle='-', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('AF8 Comparison')
plt.legend()
plt.grid(True)

# Step 7: Adjust layout for better spacing between the plots
plt.tight_layout()

# Step 8: Display the plot
plt.show()
