# PISO-framework-MIMIC-IV

## Overview

This repository implements the paper **"PiSO: Pipelined Selection Optimization Framework for Preprocessed Data and DL Model on ABP Estimation using MIMIC-IV"**.

### Dependencies
The code is written in Python and utilizes the following libraries for deep learning models:
- **TensorFlow**
- **PyTorch**

### Project Organization

- **Data Cleaning Folder**: Contains a Jupyter Notebook that extracts signals from the MIMIC-IV database with appropriate duration and ranges, and creates a local raw dataset in `.mat` format.
  
- **PPDS Folder**: Contains a Jupyter Notebook that utilizes the raw dataset and applies four different preprocessing (PP) techniques. It generates four preprocessed (PP) datasets in `.npy` format. Note that the number of preprocessing techniques can be increased or adjusted as needed.

- **DLMS Folder**: Contains five models trained on ECG-only signals to predict ABP.

### Customization

1. **Change the Preprocessed Dataset**:
   You can update the PP dataset by using the `gdown` library to download a new dataset. To do this, change the `file_id` and update the dataset URL:
   ```python
   file_id = "12KmtjeVHuZP4omk-AvEBpfwmI1cPn1R7"  # Change the file_id
   download_url = f"https://drive.google.com/uc?id={file_id}"
   output_file = "butter_maf_dataset.npy"
   gdown.download(download_url, output_file, quiet=False)

2. **Change ECG-Only Training to PPG-Only Training**:
   To switch to PPG-only training, replace the ECG signal extraction with PPG:
   bm_data = np.load("butter_maf_dataset.npy", allow_pickle=True).item()
   ecg_bm = accumulate_episodes(bm_data, signal_type='filtered_ecg')  # Use 'filtered_ppg' instead of 'filtered_ecg'

3. **Train with PPG + ECG**:
   To train with both ECG and PPG signals
   1) Extract both ECG and PPG signals.
   2) Use np.stack to combine them.
   3) Increase the input channels from 1 to 2.