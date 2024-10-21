import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

# ## Disable oneDNN optimization on intel 
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from models import CLSModel, InceptionModel, LogisticModel, SVMModel, DeepModel
from dataset import PNCC
from utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
from prettytable import PrettyTable

# Print to console using PrettyTable
table = PrettyTable()
table.field_names = ["Model Type", "Accuracy", "Precision", "Recall", "F1 Score"]

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs found. Using CPU.")
    else:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")

def process_data(data_path):
    ecg_data, combined_pcg_mel, mel, labels = PNCC(data_path).data_process()
    
    # Create a random shuffled array with test size = 30%
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
    
    # Use these indices to get the training and testing data for ecg and pcg
    ecg_train = ecg_data[train_indices]
    ecg_test = ecg_data[test_indices]
    pcg_train = combined_pcg_mel[train_indices]
    pcg_test = combined_pcg_mel[test_indices]
    mel_train = mel[train_indices]
    mel_test = mel[test_indices]
    labels_train = labels[train_indices]
    labels_test = labels[test_indices]
    
    return ecg_train, ecg_test, pcg_train, pcg_test, mel_train, mel_test, labels_train, labels_test
def train_model(model_class, train_data, labels_train, test_data, labels_test):
    model_instance = model_class()
    model, labels_pred = model_instance.train(train_data, labels_train, test_data, labels_test)
    return model, labels_pred

def train_inception_model(mel_train, labels_train, mel_test, labels_test):
    inception_model = InceptionModel()
    labels_pred_deep = inception_model.train(mel_train, labels_train, mel_test, labels_test)
    return inception_model, labels_pred_deep


def write_metrics(f, model_type, labels_test, labels_pred):
    accuracy = accuracy_score(labels_test, labels_pred)
    precision = precision_score(labels_test, labels_pred, average='binary', pos_label=1)
    recall = recall_score(labels_test, labels_pred, average='binary', pos_label=1)
    f1 = f1_score(labels_test, labels_pred, average='binary', pos_label=1)
    
    # Write to CSV file
    writer = csv.writer(f)
    writer.writerow([model_type, accuracy, precision, recall, f1])
    

    table.add_row([model_type, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
    
    return accuracy, precision, recall, f1

def train_and_evaluate(model_type, model_class, train_data, labels_train, test_data, labels_test, f):
    model, labels_pred = train_model(model_class, train_data, labels_train, test_data, labels_test)
    plot_confusion_matrix(labels_test, labels_pred, labels=["Normal", "Abnormal"], model_type=model_type)
    write_metrics(f, model_type, labels_test, labels_pred)

def main():
    setup_gpu()
    sns.set_style(style="darkgrid")
    data_path = 'training-a'
    ecg_train, ecg_test, pcg_train, pcg_test, mel_train, mel_test, labels_train, labels_test = process_data(data_path)

    # Replace any NaN values with the mean of the respective columns
    ecg_train = np.nan_to_num(ecg_train, nan=np.nanmean(ecg_train))
    ecg_test = np.nan_to_num(ecg_test, nan=np.nanmean(ecg_test))
    pcg_train = np.nan_to_num(pcg_train, nan=np.nanmean(pcg_train))
    pcg_test = np.nan_to_num(pcg_test, nan=np.nanmean(pcg_test))

    # Open a file to write the accuracies
    with open('model_accuracies.csv', 'a') as f:
        f.write("\n")
        
        # train_and_evaluate("InceptionModel (ECG only)", InceptionModel, ecg_train, labels_train, ecg_test, labels_test, f)
        # Training for ECG only with RandomForest
        train_and_evaluate("DeepModel (ECG only)", DeepModel, ecg_train, labels_train, ecg_test, labels_test, f)
        train_and_evaluate("RandomForest (ECG only)", CLSModel, ecg_train, labels_train, ecg_test, labels_test, f)

        # # Training for ECG only with Logistic
        # train_and_evaluate("Logistic (ECG only)", LogisticModel, ecg_train, labels_train, ecg_test, labels_test, f)
        
        # # Training for ECG only with SVM
        # train_and_evaluate("SVM (ECG only)", SVMModel, ecg_train, labels_train, ecg_test, labels_test, f)

        # Training for PCG only with RandomForest
        train_and_evaluate("DeepModel (PCG only)", DeepModel, pcg_train, labels_train, pcg_test, labels_test, f)
        train_and_evaluate("RandomForest (PCG only)", CLSModel, pcg_train, labels_train, pcg_test, labels_test, f)

        # # Training for PCG only with Logistic
        # train_and_evaluate("Logistic (PCG only)", LogisticModel, pcg_train, labels_train, pcg_test, labels_test, f)

        # # Training for PCG only with SVM
        # train_and_evaluate("SVM (PCG only)", SVMModel, pcg_train, labels_train, pcg_test, labels_test, f)

        # Training for both ECG and PCG with RandomForest
        combine_train = np.concatenate((ecg_train, pcg_train), axis=1)
        combine_test = np.concatenate((ecg_test, pcg_test), axis=1)
        
        # Training for both ECG and PCG with RandomForest
        train_and_evaluate("DeepModel (ECG + PCG)", DeepModel, combine_train, labels_train, combine_test, labels_test, f)
        train_and_evaluate("RandomForest (ECG + PCG)", CLSModel, combine_train, labels_train, combine_test, labels_test, f)

        # # Training for both ECG and PCG with Logistic
        # train_and_evaluate("Logistic (ECG + PCG)", LogisticModel, combine_train, labels_train, combine_test, labels_test, f)

        # # Training for both ECG and PCG with SVM
        # train_and_evaluate("SVM (ECG + PCG)", SVMModel, combine_train, labels_train, combine_test, labels_test, f)


if __name__ == "__main__":
    main()
    print(table)

    # inception_model.save('model.h5')