import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from models import train_cls_model, InceptionModel
from dataset import PNCC
from utils import plot_confusion_matrix

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        tf.config.set_visible_devices(gpus[0],'GPU')

    # Set seaborn style
    sns.set(style="darkgrid")

    # Set data path
    data_path = 'training-a'
    combine, mel, labels = PNCC(data_path).data_process()

    # Split training and testing sets
    combine_train, combine_test, labels_train, labels_test = train_test_split(combine, labels, test_size=0.3, random_state=42)
    mel_train, mel_test = train_test_split(mel, test_size=0.3, random_state=42)

    cls_model, labels_pred_cls = train_cls_model(combine_train, labels_train, combine_test, labels_test)

    input_shape = (224, 224, 3)  # Input for InceptionV3
    inception_model = InceptionModel(input_shape)
    labels_pred_deep = inception_model.train(mel_train, labels_train, mel_test, labels_test)

    # Plot confusion matrices for both models
    plot_confusion_matrix(labels_test, labels_pred_cls, labels=["Normal", "Abnormal"], model_type="RandomForest")
    plot_confusion_matrix(labels_test, labels_pred_deep, labels=["Normal", "Abnormal"], model_type="InceptionV3")
    inception_model.save('model.h5')

if __name__ == "__main__":
    main()