import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, convolve
from scipy.ndimage import binary_opening, binary_closing

# Assuming these are your pre-existing Python functions
# from your_module import import_analysis_file, read_binary_iq_data, get_burst_indices
def import_analysis_file(file_path):
    """
    Reads a CSV file and returns a dictionary of lists
    """
    # Define your hardcoded mapping: "Original Header": "snake_case_key"
    header_mapping = {
        "Frame #": "frame_number",
        "Frame ID": "frame_id",
        "Start time s": "start_time",
        "Frame length": "frame_length",
		"sample_rate": "sample_rate",
		"Filename": "filename"
    }
    
    # Initialize the output dictionary with empty lists
    data_dict = {key: [] for key in header_mapping.values()}
    
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} was not found.")
        return None

    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as csv_file:
            # DictReader uses the first row as keys
            reader = csv.DictReader(csv_file)
            
            for row in reader:
                for original_header, snake_key in header_mapping.items():
                    # Append the value from the row to the corresponding list
                    data_dict[snake_key].append(row.get(original_header))
                    
        return data_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        

def get_burst_indices(iq_snippet, t_snippet, threshold, sample_rate):
    # Calculate magnitude and sample indices
    iq_mag = np.abs(iq_snippet)
    sample_index = np.arange(len(iq_mag))

    # Smooth the data to reduce highs and lows (Moving Average Filter)
    filter_size = 31
    filter_kernel = np.ones(filter_size) / filter_size
    iq_mag_filt = np.convolve(iq_mag, filter_kernel, mode='same')
    
    # Calculate the threshold
    iq_mag_filt_mean = threshold * np.mean(iq_mag_filt)
    
    # Create a binary amplitude map
    iq_mag_bin = iq_mag_filt > iq_mag_filt_mean

    iq_mag_bin_sum = np.sum(iq_mag_bin)
    iq_mag_sum_delta = 1e6

    num_contiguous_samples = int(np.ceil(sample_rate * 20e-6) + 1)
    count = 0
    min_pulse_width = int(np.ceil(sample_rate * 1e-6) + 1)

    # Run an open operation to remove small blips (using a 1D structure element)
    iq_mag_bin = binary_opening(iq_mag_bin, structure=np.ones(7)).astype(int)

    # Iterative closing to fill gaps
    while iq_mag_sum_delta > 1 and count < 100:
        iq_mag_bin = binary_closing(iq_mag_bin, structure=np.ones(num_contiguous_samples)).astype(int)
        
        current_sum = np.sum(iq_mag_bin)
        iq_mag_sum_delta = np.abs(iq_mag_bin_sum - current_sum)
        iq_mag_bin_sum = current_sum
        count += 1
    
    # Final open to remove any remaining narrow pulses
    iq_mag_bin = binary_opening(iq_mag_bin, structure=np.ones(min_pulse_width)).astype(int)

    # Convert binary [0, 1] to [-1, 1] for correlation/transition detection
    iq_mag_bin_bipolar = 2 * iq_mag_bin - 1
    
    # Find the transitions using correlation (convolution with flipped kernel)
    # MATLAB's 'same' convolution centers the kernel
    start_filter = np.array([-1, 1, 1])
    stop_filter = np.array([1, 1, -1])
    
    # In Python convolve, we flip the filter to match MATLAB's conv behavior
    start_corr = np.convolve(iq_mag_bin_bipolar, start_filter[::-1], mode='same')
    stop_corr = np.convolve(iq_mag_bin_bipolar, stop_filter[::-1], mode='same')
    
    # Identify indices where the filter perfectly matches (sum = 3)
    start_index = sample_index[start_corr == 3]
    stop_index = sample_index[stop_corr == 3]
    
    # Add point for start if the first element is active
    if iq_mag_bin[0] == 1:
        start_index = np.insert(start_index, 0, 0)

    # Add point for stop if the last element is active
    if iq_mag_bin[-1] == 1:
        stop_index = np.append(stop_index, len(iq_mag_bin) - 1)

    # Align lengths
    min_len = min(len(stop_index), len(start_index))
    start_index = start_index[:min_len]
    stop_index = stop_index[:min_len]

    # Calculate lengths and off-times
    burst_lengths = stop_index - start_index
    off_times = start_index[1:] - stop_index[:-1]

    indices = np.column_stack((start_index, stop_index))

    # Plotting
    plt.figure(1001)
    plt.plot(t_snippet, iq_mag_filt, 'g', label='Filtered IQ')
    plt.axhline(y=iq_mag_filt_mean, color='k', linestyle='--', label='Threshold')
    plt.plot(t_snippet, iq_mag_bin, 'b', label='Binary Map')

    if len(indices) > 0:
        plt.scatter(t_snippet[indices[:, 0]], np.ones(len(indices)), c='r', marker='o', label='Start')
        plt.scatter(t_snippet[indices[:, 1]], np.ones(len(indices)), c='k', marker='o', label='Stop')
    
    plt.legend()
    plt.show()

    return burst_lengths, off_times, indices
    
    
def run_analysis():
    # Setup formatting (similar to format long g)
    np.set_printoptions(precision=15, suppress=True)

    # Initial paths
    iq_filepath = r'D:\Projects\data\RF\20240720\RB\5M\\'
    
    # --- load in data ---
    # Note: Python doesn't have a direct equivalent to uigetfile without tkinter
    # Placeholder for file selection logic
    data_filepath = r'D:\Projects\data\RF\20240720\\'
    data_filename_raw = input("Enter the CSV filename: ")
    data_filename = os.path.join(data_filepath, data_filename_raw)

    if not os.path.exists(data_filename):
        print("File not found.")
        return

    # Non-standard function call
    # [iq_filename, sample_rate, frame_number, frame_id, start_times, frame_lengths]
    results = import_analysis_file(data_filename, start_row=5)
    iq_filename, sample_rate, frame_num, frame_id, start_times, frame_lengths = results

    # --- determine scaling ---
    file_ext = os.path.splitext(iq_filename)[1]
    if file_ext == '.fc32':
        scale = 32768 / 2048
        dtype = np.complex64 # Assuming complex for fc32
    else:
        scale = 1 / 2048
        dtype = np.int16

    # --- read in iq data ---
    # Non-standard function call
    _, iqc_in = read_binary_iq_data(os.path.join(iq_filepath, iq_filename), dtype=dtype)
    
    # Scale and create complex array if necessary
    iqc = iqc_in.astype(np.complex128) * scale
    
    # Time vector
    t = np.arange(len(iqc)) / sample_rate

    start_samples = np.floor(sample_rate * (start_times - 0.002)).astype(int)
    frame_length_samples = np.ceil(sample_rate * frame_lengths).astype(int)

    # --- Frame ID input ---
    frame_index_input = input("Frame Index: ")
    frame_index = int(frame_index_input)
    print(f"frame index: {frame_index}")

    burst_lengths = []
    off_times = []
    sample_indices = []

    # --- Burst Detection Loop ---
    for idx in range(len(frame_id)):
        if frame_id[idx] == frame_index:
            print(f"index: {idx}")

            start_idx = start_samples[idx]
            end_idx = start_idx + frame_length_samples[idx]
            
            iq_snippet = iqc[start_idx:end_idx]
            t_snippet = t[start_idx:end_idx]
            
            # Non-standard function call: get_burst_indices
            bl, ot, indices = get_burst_indices(iq_snippet, t_snippet, 1.28, sample_rate)

            burst_lengths.append(bl)
            off_times.append(ot)
            sample_indices.append(indices)

    print("frame ID complete")

    # --- Mean Calculation Logic ---
    # Python Note: Handling nested lists/arrays for 'burst_mean'
    def calculate_custom_mean(data_list):
        if not data_list: return np.array([])
        
        # Convert list of varying lengths to a padded numpy array if necessary
        # This mirrors the MATLAB logic of finding most common edges
        # Simplified here to provide the mean of the collected data
        arr = np.array(data_list, dtype=object)
        # ... logic for histogram/edge detection ...
        return np.mean(arr, axis=0) # Simplified placeholder

    # Note: The original MATLAB code used a very specific histogram-based 
    # mode finder. This is simplified to a standard mean for brevity.
    # If specific edge detection is needed, use np.histogram.

    # --- Modulation Analysis ---
    for kdx in range(len(frame_id)):
        if frame_id[kdx] == frame_index:
            
            s_time = start_samples[kdx]
            e_time = s_time + frame_length_samples[kdx] + int(sample_rate * 110e-6)
            
            iq_snippet = iqc[s_time:e_time]
            t_snippet = t[s_time:e_time]

            # Figure 100: Amplitude
            plt.figure(100)
            plt.plot(t_snippet, np.abs(iq_snippet), 'g')
            plt.title(f"Index: {kdx}")
            plt.show(block=False)

            # Figure 101: Spectrogram
            plt.figure(101)
            f, t_spec, Sxx = signal.spectrogram(iq_snippet, sample_rate, 
                                                window='hann', nperseg=1024, 
                                                noverlap=1000, detrend=False)
            plt.pcolormesh(t_spec, np.fft.fftshift(f), np.fft.fftshift(10 * np.log10(Sxx), axes=0))
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show(block=False)

            # Loop through individual bursts
            for idx_burst in range(len(sample_indices)):
                for jdx in range(len(sample_indices[idx_burst])):
                    
                    indices = sample_indices[idx_burst][jdx]
                    burst_start = start_samples[kdx] + int(indices[0])
                    burst_stop = start_samples[kdx] + int(indices[1]) + int(sample_rate * 110e-6)
                    
                    iq_burst = iqc[burst_start:burst_stop]
                    t_burst = t[burst_start:burst_stop]

                    # Figure 102: I and Q components
                    plt.figure(102, figsize=(14, 5))
                    plt.plot(t_burst, i_burst.real, 'b', label='I')
                    plt.plot(t_burst, i_burst.imag, 'r', label='Q')
                    plt.legend()
                    plt.show(block=False)

                    # Figure 103: 3D Scatter (using Matplotlib mplot3d)
                    fig = plt.figure(103, figsize=(14, 5))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    num_samples = len(iq_burst)
                    lower = int(0.48 * num_samples)
                    upper = min(lower + 5000, num_samples)

                    ax.scatter(t_burst[lower:upper], i_burst[lower:upper].real, 
                               i_burst[lower:upper].imag, c='b', marker='o')
                    ax.plot(t_burst[lower:upper], i_burst[lower:upper].real, 
                            i_burst[lower:upper].imag, 'b--')
                    
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('I')
                    ax.set_zlabel('Q')
                    plt.show()

                    input("Press Enter for next burst...")

    print("Modulation complete")

if __name__ == "__main__":
    run_analysis()
	
	
	
	
	
import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QFileDialog)

class FileSelectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('File Path Selector')
        self.setGeometry(300, 300, 400, 200)
        
        layout = QVBoxLayout()

        # IQ File Path Section
        self.iq_label = QLabel('IQ File Path: Not selected')
        self.iq_btn = QPushButton('Select IQ File Path')
        self.iq_btn.clicked.connect(self.get_iq_filepath)
        
        # Data Filename Section
        self.data_label = QLabel('Data Filename: Not selected')
        self.data_btn = QPushButton('Select Data Filename')
        self.data_btn.clicked.connect(self.get_data_filename)

        # Add widgets to layout
        layout.addWidget(self.iq_label)
        layout.addWidget(self.iq_btn)
        layout.addSpacing(20) # Visual gap
        layout.addWidget(self.data_label)
        layout.addWidget(self.data_btn)

        self.setLayout(layout)

    def get_iq_filepath(self):
        # getOpenFileName returns a tuple (path, selected_filter)
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select IQ File", 
            "", 
            "IQ Files (*.iq *.dat);;All Files (*)"
        )
        if file_path:
            self.iq_label.setText(f"IQ File Path: {file_path}")
            print(f"Stored IQ Path: {file_path}")

    def get_data_filename(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Data File", 
            "", 
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            # If you only want the filename (not the full path), use os.path.basename
            import os
            filename = os.path.basename(file_path)
            self.data_label.setText(f"Data Filename: {filename}")
            print(f"Stored Data Filename: {filename}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileSelectorApp()
    ex.show()
    sys.exit(app.exec())