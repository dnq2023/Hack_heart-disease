import os
import numpy as np
import pandas as pd
import librosa
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import Input

# Set seaborn style
sns.set(style="darkgrid")

# Set data path
data_path = 'training-a'
wav_files = sorted([f for f in os.listdir(data_path) if f.endswith('.wav')])
dat_files = sorted([f for f in os.listdir(data_path) if f.endswith('.dat')])

# Read REFERENCE-SQI.csv
labels_file = pd.read_csv(os.path.join(data_path, 'REFERENCE-SQI.csv'), header=None,
                          names=['record_name', 'label', 'reliability'])


# Extract PCG signal and convert to Mel spectrogram
def extract_pcg_mel_spectrogram(wav_file, sr=2000, n_mels=128, fmax=8000):
    wav_data, _ = librosa.load(os.path.join(data_path, wav_file), sr=sr)
    S = librosa.feature.melspectrogram(y=wav_data, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


# Extract ECG signal features
def extract_ecg_features(dat_file):
    record = wfdb.rdrecord(os.path.join(data_path, dat_file.split('.')[0]))
    ecg_signal = record.p_signal[:, 0]
    diff_ecg = np.diff(ecg_signal)
    mean_ecg = np.mean(ecg_signal)
    std_ecg = np.std(ecg_signal)
    features = np.array([mean_ecg, std_ecg, np.mean(diff_ecg), np.std(diff_ecg)])
    return features


# Plot signal images
def plot_signals(ecg_signal, pcg_signal, ecg_features, pcg_features, record_name):
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    # Plot ECG signal
    axs[0, 0].plot(ecg_signal, color='cyan')
    axs[0, 0].set_title(f'ECG Signal for {record_name}', fontsize=14)

    # Plot PCG signal
    axs[0, 1].plot(pcg_signal, color='lime')
    axs[0, 1].set_title(f'PCG Signal for {record_name}', fontsize=14)

    # Plot ECG features
    axs[1, 0].plot(ecg_features, color='blue')
    axs[1, 0].set_title(f'ECG Features for {record_name}', fontsize=14)

    # Plot PCG features
    axs[1, 1].plot(pcg_features, color='green')
    axs[1, 1].set_title(f'PCG Features for {record_name}', fontsize=14)

    # Plot ECG spectrum
    ecg_spectrum = np.abs(np.fft.fft(ecg_signal))[:len(ecg_signal) // 2]
    axs[2, 0].plot(ecg_spectrum, color='red')
    axs[2, 0].set_title(f'ECG Frequency Spectrum for {record_name}', fontsize=14)

    # Plot PCG spectrum
    pcg_spectrum = np.abs(np.fft.fft(pcg_signal))[:len(pcg_signal) // 2]
    axs[2, 1].plot(pcg_spectrum, color='magenta')
    axs[2, 1].set_title(f'PCG Frequency Spectrum for {record_name}', fontsize=14)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{record_name}_signals.png')
    plt.show()


# Data loading and feature extraction
X = []
X_mel = []  # For storing Mel spectrograms
y = []
for wav_file, dat_file in zip(wav_files, dat_files):
    record_name = wav_file.split('.')[0]
    record_info = labels_file[labels_file['record_name'] == record_name]

    # Check data reliability
    if not record_info.empty and record_info['reliability'].values[0] == 1:
        label = record_info['label'].values[0]
        pcg_mel_spectrogram = extract_pcg_mel_spectrogram(wav_file)  # Extract PCG Mel spectrogram
        ecg_features = extract_ecg_features(dat_file)

        # Plot signal images
        plot_signals(ecg_features, pcg_mel_spectrogram.flatten(), ecg_features, np.mean(pcg_mel_spectrogram, axis=1), record_name)

        combined_features = np.hstack([np.mean(pcg_mel_spectrogram), ecg_features])  # Simplify features for simple model
        X.append(combined_features)
        X_mel.append(np.stack([np.resize(pcg_mel_spectrogram, (224, 224))]*3, axis=-1))  # Reshape to 224x224x3 format
        y.append(1 if label == 1 else 0)

# Convert to arrays
X = np.array(X)
X_mel = np.array(X_mel)
y = np.array(y)

# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_mel, X_test_mel = train_test_split(X_mel, test_size=0.3, random_state=42)


# Step 1: CLS model (simple model)
def train_cls_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print('Random Forest Accuracy:', accuracy_score(y_test, rf_pred))
    return rf, rf_pred


cls_model, y_pred_cls = train_cls_model(X_train, y_train, X_test, y_test)


# Step 2: BiLSTM-GoogLeNet-DS (deep model)
# Modified model input
def build_deep_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Explicit input layer
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_inception_model(input_shape):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Deep learning model training
def deep_model_pipeline(X_train_mel, y_train, X_test_mel, y_test):
    input_shape = (224, 224, 3)  # Input for InceptionV3
    inception_model = build_inception_model(input_shape)
    inception_model.fit(X_train_mel, y_train, epochs=32, batch_size=16, validation_split=0.2)

    inception_acc = inception_model.evaluate(X_test_mel, y_test)
    y_pred_deep = (inception_model.predict(X_test_mel) > 0.5).astype("int32")
    print('InceptionV3 Accuracy:', inception_acc)
    return y_pred_deep


y_pred_deep = deep_model_pipeline(X_train_mel, y_train, X_test_mel, y_test)


# Generate confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, model_type=""):
    os.makedirs('results', exist_ok=True)

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix ({model_type})', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save confusion matrix image
    plt.savefig(f'results/confusion_matrix_{model_type}.png')
    plt.show()


# Plot confusion matrices for both models
plot_confusion_matrix(y_test, y_pred_cls, labels=["Normal", "Abnormal"], model_type="RandomForest")
plot_confusion_matrix(y_test, y_pred_deep, labels=["Normal", "Abnormal"], model_type="InceptionV3")
