import os
import wfdb
import numpy as np
import biosppy
from scipy.signal import find_peaks, butter, lfilter
import seaborn as sns

class ECGProcessor:
    def __init__(self, data_path, dat_file, sampling_rate=1000):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.ecg_signal = self.load_and_filter_ecg(dat_file)
        self.r_peaks = []
        self.pqrst_waves = []
        self.detect_pqrst_waves()

    def load_and_filter_ecg(self, dat_file):
        record = wfdb.rdrecord(os.path.join(self.data_path, dat_file.split('.')[0]))
        ecg_signal = record.p_signal[:, 0]
        ecg_signal = self.apply_notch_filter(ecg_signal)
        ecg_signal = self.apply_bandpass_filter(ecg_signal)
        return ecg_signal

    def apply_notch_filter(self, signal, notch_freq=50, quality_factor=30):
        b, a = butter(2, [notch_freq - 1, notch_freq + 1], btype='bandstop', fs=self.sampling_rate)
        return lfilter(b, a, signal)

    def apply_bandpass_filter(self, signal, lowcut=5, highcut=15, order=5):
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, signal)

    def detect_pqrst_waves(self):
        out = biosppy.signals.ecg.ecg(signal=self.ecg_signal, sampling_rate=self.sampling_rate, show=False)
        self.r_peaks = out['rpeaks']
        self.pqrst_waves = [self.ecg_signal[max(0, r - int(0.2 * self.sampling_rate)):min(len(self.ecg_signal), r + int(0.4 * self.sampling_rate))] for r in self.r_peaks]

    def compute_average_pqrst_wave(self):
        if not self.pqrst_waves:
            raise ValueError("No PQRST waves detected. Please run detect_pqrst_waves() first.")
        max_length = max(len(wave) for wave in self.pqrst_waves)
        padded_waves = np.array([np.pad(wave, (0, max_length - len(wave)), 'constant') for wave in self.pqrst_waves])
        return np.mean(padded_waves, axis=0)

    def compute_r_peak_count(self):
        return len(self.r_peaks)

    def compute_average_rr_interval(self):
        rr_intervals = np.diff(self.r_peaks) / self.sampling_rate
        return np.mean(rr_intervals) if rr_intervals.size > 0 else 0

    def compute_average_qrs_width(self):
        qrs_widths = [(min(len(self.ecg_signal), r + int(0.1 * self.sampling_rate)) - max(0, r - int(0.1 * self.sampling_rate))) / self.sampling_rate for r in self.r_peaks]
        return np.mean(qrs_widths) if qrs_widths else 0

    def compute_average_st_interval(self):
        st_intervals = [(self.r_peaks[i + 1] - (self.r_peaks[i] + int(0.1 * self.sampling_rate))) / self.sampling_rate for i in range(len(self.r_peaks) - 1)]
        return np.mean(st_intervals) if st_intervals else 0

    def compute_heart_rate(self):
        rr_intervals = np.diff(self.r_peaks) / self.sampling_rate
        heart_rate = 60 / rr_intervals if rr_intervals.size > 0 else 0
        return np.mean(heart_rate)

    def compute_sdnn(self):
        rr_intervals = np.diff(self.r_peaks) / self.sampling_rate
        return np.std(rr_intervals) if rr_intervals.size > 0 else 0

    def compute_rmssd(self):
        rr_intervals = np.diff(self.r_peaks) / self.sampling_rate
        diff_rr_intervals = np.diff(rr_intervals)
        return np.sqrt(np.mean(diff_rr_intervals**2)) if diff_rr_intervals.size > 0 else 0

    def extract_features(self):

        features = {
            'number_of_r_peaks': self.compute_r_peak_count(),
            'average_rr_interval': self.compute_average_rr_interval(),
            'average_qrs_width': self.compute_average_qrs_width(),
            'average_st_interval': self.compute_average_st_interval(),
            'heart_rate': self.compute_heart_rate(),
            'sdnn': self.compute_sdnn(),
            'rmssd': self.compute_rmssd()
        }

        feature_values = np.array(list(features.values()))
        
        return features, feature_values


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Example usage:
    ecg_processor = ECGProcessor(data_path="training-a", dat_file="a0001.dat")
    print(ecg_processor.extract_features())

    # Plot the average PQRST wave using seaborn
    plt.figure(figsize=(10, 4))
    # sns.lineplot(x=range(len(ecg_signal)), y=ecg_signal)
    sns.lineplot(x=range(len(ecg_processor.compute_average_pqrst_wave())), y=ecg_processor.compute_average_pqrst_wave())
    plt.title('Average PQRST Wave')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()
