from scipy.signal import hilbert, iirnotch, filtfilt, butter
import wfdb
import os
import numpy as np
import pywt
import librosa
import pandas as pd


class PNCC():

    def __init__(self, data_path) -> None:
        self.data_path = data_path


    # Apply notch filter to remove powerline noise (50 Hz)
    def apply_notch_filter(self, signal, fs=1000, freq=50, Q=30):
        w0 = freq / (fs / 2)
        b, a = iirnotch(w0, Q)
        return filtfilt(b, a, signal)

    # High-pass and low-pass filters for frequency selection
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, signal, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        return filtfilt(b, a, signal)

    # Extract ECG features using basic Pan-Tompkins algorithm
    def extract_ecg_features(self, dat_file, fs=1000):
        record = wfdb.rdrecord(os.path.join(self.data_path, dat_file.split('.')[0]))
        ecg_signal = record.p_signal[:, 0]

        # Notch filter for 50 Hz powerline noise
        ecg_filtered = self.apply_notch_filter(ecg_signal, fs)

        # Bandpass filter (5-15 Hz) to extract R-peaks
        ecg_filtered = self.bandpass_filter(ecg_filtered, 5, 15, fs)

        # Compute simple statistical features
        diff_ecg = np.diff(ecg_filtered)
        mean_ecg = np.mean(ecg_filtered)
        std_ecg = np.std(ecg_filtered)
        return np.array([mean_ecg, std_ecg, np.mean(diff_ecg), np.std(diff_ecg)])

    # Extract PCG signal features using wavelet transform and Hilbert transform
    def extract_pcg_features(self, wav_file, sr=1000):
        wav_data, _ = librosa.load(os.path.join(self.data_path, wav_file), sr=sr)

        # Wavelet transform for multi-scale feature extraction
        coeffs = pywt.wavedec(wav_data, 'db4', level=5)
        cA5 = coeffs[0]

        # Hilbert transform for envelope detection
        analytic_signal = hilbert(cA5)
        amplitude_envelope = np.abs(analytic_signal)

        # Compute basic statistical features
        mean_envelope = np.mean(amplitude_envelope)
        std_envelope = np.std(amplitude_envelope)

        return np.array([mean_envelope, std_envelope])

    # Extract PCG signal and convert to Mel spectrogram
    def extract_pcg_mel_spectrogram(self, wav_file, sr=2000, n_mels=128, fmax=8000):
        wav_data, _ = librosa.load(os.path.join(self.data_path, wav_file), sr=sr)
        S = librosa.feature.melspectrogram(y=wav_data, sr=sr, n_mels=n_mels, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB


    def data_process(self):   
        wav_files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.wav')])
        dat_files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.dat')])

        # Read REFERENCE-SQI.csv
        labels_file = pd.read_csv(os.path.join(self.data_path, 'REFERENCE-SQI.csv'), header=None,
                                names=['record_name', 'label', 'reliability'])


        # Data loading and feature extraction
        combine = []
        mel = []  # For storing Mel spectrograms
        labels = []
        for idx, (wav_file, dat_file) in enumerate(zip(wav_files, dat_files)):
            record_name = wav_file.split('.')[0]
            record_info = labels_file[labels_file['record_name'] == record_name]

            # Check data reliability
            if not record_info.empty and record_info['reliability'].values[0] == 1:
                label = record_info['label'].values[0]
                pcg_mel_spectrogram = self.extract_pcg_mel_spectrogram(wav_file)  # Extract PCG Mel spectrogram
                ecg_features = self.extract_ecg_features(dat_file)
                pcg_features = self.extract_pcg_features(wav_file)

                combined_features = np.hstack([np.mean(pcg_mel_spectrogram), ecg_features])  # Simplify features for simple model
                combine.append(combined_features)
                mel.append(np.stack([np.resize(pcg_mel_spectrogram, (224, 224))]*3, axis=-1))  # Reshape to 224x224x3 format
                labels.append(1 if label == 1 else 0)

        # Convert to arrays
        combine = np.array(combine)
        mel = np.array(mel)
        labels = np.array(labels)
        return combine, mel, labels

##################
# The default sr is not the same for the 2 pcg functions 
# the pcg features are not used for training