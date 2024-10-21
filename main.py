import seaborn as sns
from sklearn.model_selection import train_test_split

# ## Disable oneDNN optimization on intel 
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from models import CLSModel, InceptionModel
from dataset import PNCC
from utils import plot_confusion_matrix

def main():
    gpus = tf.config.list_physical_devices('GPU')
    # Check if there are any GPUs available
    if not gpus:
        print("No GPUs found. Using CPU.")
    else:
        # If GPUs are available, set the first GPU as visible
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")

    # Set seaborn style
    sns.set_style(style="darkgrid")

    # Set data path
    data_path = 'training-a'
    combine, mel, labels = PNCC(data_path).data_process()

    # Split training and testing sets
    combine_train, combine_test, labels_train, labels_test = train_test_split(combine, labels, test_size=0.3, random_state=42)
    mel_train, mel_test = train_test_split(mel, test_size=0.3, random_state=42)

    cls_model, labels_pred_cls = CLSModel().train(combine_train, labels_train, combine_test, labels_test)

    input_shape = (224, 224, 3)  # Input for InceptionV3
    inception_model = InceptionModel(input_shape)
    labels_pred_deep = inception_model.train(mel_train, labels_train, mel_test, labels_test)

    # Plot confusion matrices for both models
    plot_confusion_matrix(labels_test, labels_pred_cls, labels=["Normal", "Abnormal"], model_type="RandomForest")
    plot_confusion_matrix(labels_test, labels_pred_deep, labels=["Normal", "Abnormal"], model_type="InceptionV3")
    inception_model.save('model.h5')

if __name__ == "__main__":
    main()