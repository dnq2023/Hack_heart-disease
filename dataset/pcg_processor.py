import os
import numpy as np
import librosa
import librosa.display
from scipy.signal import find_peaks
import seaborn as sns

class PCGProcessor:
    def __init__(self, data_path, wav_file, sampling_rate=22050):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.pcg_signal, self.sampling_rate = self.load_and_filter_pcg(wav_file)
        self.features = self.extract_features()

    def load_and_filter_pcg(self, wav_file):
        file_path = os.path.join(self.data_path, wav_file)
        pcg_signal, sr = librosa.load(file_path, sr=self.sampling_rate)
        return pcg_signal, sr

    def extract_features(self):
        features = {}
        features['mfccs'] = self.compute_mfccs()
        features['zcr'] = self.compute_zcr()
        features['spectral_centroid'] = self.compute_spectral_centroid()
        features['rms'] = self.compute_rms()
        features['spectral_bandwidth'] = self.compute_spectral_bandwidth()
        features['spectral_rolloff'] = self.compute_spectral_rolloff()
        features['chroma_stft'] = self.compute_chroma_stft()
        
        feature_keys = list(features.keys())
        feature_values = np.array(list(features.values()))
        
        return feature_keys, feature_values

    def compute_mfccs(self, n_mfcc=13):
        mfccs = librosa.feature.mfcc(y=self.pcg_signal, sr=self.sampling_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs)

    def compute_zcr(self):
        zcr = librosa.feature.zero_crossing_rate(y=self.pcg_signal)
        return np.mean(zcr)

    def compute_spectral_centroid(self):
        spectral_centroid = librosa.feature.spectral_centroid(y=self.pcg_signal, sr=self.sampling_rate)
        return np.mean(spectral_centroid)

    def compute_rms(self):
        rms = librosa.feature.rms(y=self.pcg_signal)
        return np.mean(rms)

    def compute_spectral_bandwidth(self):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.pcg_signal, sr=self.sampling_rate)
        return np.mean(spectral_bandwidth)

    def compute_spectral_rolloff(self):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.pcg_signal, sr=self.sampling_rate)
        return np.mean(spectral_rolloff)

    def compute_chroma_stft(self):
        chroma_stft = librosa.feature.chroma_stft(y=self.pcg_signal, sr=self.sampling_rate)
        return np.mean(chroma_stft)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Example usage:
    pcg_processor = PCGProcessor(data_path="training-a", wav_file="a0001.wav")
    print(pcg_processor.extract_features())

    # Plot the average PQRST wave using seaborn
    # plt.figure(figsize=(10, 4))
    # sns.lineplot(x=range(len(ecg_signal)), y=ecg_signal)
    # # sns.lineplot(x=range(len(pcg_processor.compute_average_pqrst_wave())), y=pcg_processor.compute_average_pqrst_wave())
    # plt.title('Average PQRST Wave')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.show()

# Example usage:
# pcg_processor = PCGProcessor(data_path='/path/to/data', wav_file='example.wav')
# features = pcg_processor.features
# print(features)
