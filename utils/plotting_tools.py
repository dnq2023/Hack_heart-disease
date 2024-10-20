import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


# Plot signal images ******change the length of the axis
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
    # plt.show()


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
    # plt.show()


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
    # plt.show()