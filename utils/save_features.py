"""
The code is largely from the paper of Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, I. (2017).
"Vision-Based Fall Detection with Convolutional Neural Networks" Wireless Communications and Mobile Computing, 2017.
Particularly, this helped with early experiment that only trained the classifier layer of our model.
"""

from utils.utils import *
from proto.DataGenerator import *
import h5py
import gc


def save_Features(feature_extractor,
                  features_file,
                  labels_file,
                  features_key,
                  labels_key,
                  exp_name,
                  num_features):
    """
    Function to load the RGB data, do a feed-forward through the feature extractor and store the
    output feature vectors in the file 'features_file' and the  labels in 'labels_file'.
    Input:
    * feature_extractor: model without the top layer, which is the classifer
    * features_file: path to the hdf5 file where the extracted features are going to be stored
    * labels_file: path to the hdf5 file where the labels of the features are going to be stored
    * features_key: name of the key for the hdf5 file to store the features
    * labels_key: name of the key for the hdf5 file to store the labels
    """

    # Fill the IDs and classes arrays
    classes = list_data(exp_name)
    class0, class1 = classes['fail'], classes['pass']
    num_samples = len(class0) + len(class1)
    num_features[0] = num_samples

    # File to store the extracted features and datasets to store them
    # IMPORTANT NOTE: 'w' mode totally erases previous data
    # Write files in h5py format
    h5features = h5py.File(features_file, 'w')
    h5labels = h5py.File(labels_file, 'w')

    # Create data sets
    dataset_features = h5features.create_dataset(features_key,
                                                 shape=num_features,
                                                 dtype='float64')
    dataset_labels = h5labels.create_dataset(labels_key,
                                             shape=(num_samples, 1),
                                             dtype='float64')

    # Process class 0
    gen = DataGenerator(class0, np.zeros(len(class0)), 1)
    for i in range(len(class0)):
        rgb_images, rgb_labels = gen.__getitem__(i)
        predictions = feature_extractor.predict(rgb_images)

        dataset_features[i, :] = predictions
        dataset_labels[i, :] = 0

        del rgb_images, rgb_labels
        gc.collect()

    # Process class 1
    gen = DataGenerator(class1, np.ones(len(class1)), 1)
    for i in range(len(class0), num_samples):
        rgb_images, rgb_labels = gen.__getitem__(i)
        prediction = feature_extractor.predict(rgb_images)

        dataset_features[i, :] = predictions
        dataset_labels[i, :] = 1

        del rgb_images, rgb_labels
        gc.collect()

    h5features.close()
    h5labels.close()