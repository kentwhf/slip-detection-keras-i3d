from DataGenerator import DataGenerator
from i3d_inception import *

import tensorflow as tf
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
import os
import h5py
import gc
from keras.backend.tensorflow_backend import clear_session, get_session

matplotlib.use('Agg')


def list_data(exp_name, experiments):
    """
    Function to list existing data set by filename and label
    """
    data_folder = r'C:\Users\kentw\OneDrive - University of Toronto\PycharmProjects\Fall-D' \
                  r'etection-with-CNNs-and-Optical-Flow\MAA'

    vids = {'pass': [], 'fail': []}
    data_folder = [os.path.join(data_folder, category) for category in experiments[exp_name]]
    for path in data_folder:
        for file in os.listdir(path):
            if not file.lower().endswith('.npy'):
                continue
            else:
                file_dir = os.path.join(path, file)
                ID = file_dir.split('\\')[-1].rstrip('.npy')
                # Classify
                result = ID.split('_')[2][1]
                if result == "P":
                    vids["pass"].append(ID)
                else:
                    vids["fail"].append(ID)
    return vids


def extract_subID(list_):
    res = []
    for item in list_:
        sub = item.split('_')[1]
        res.append(sub)
    return res


def saveFeatures(feature_extractor,
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


# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()

    try:
        del model  # this is from global space - change this as you need
    except:
        pass

    # print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))


def build_model(weights="rgb_imagenet_and_kinetics",
                NUM_FRAMES=64,
                FRAME_HEIGHT=224,
                FRAME_WIDTH=224,
                NUM_RGB_CHANNELS=3,
                NUM_CLASSES=2,
                dropout_prob=0.36):
    # I3D ARCHITECTURE
    model = Inception_Inflated3d(
        include_top=False,
        weights=weights,
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
        classes=NUM_CLASSES)

    # ==================== CLASSIFIER ========================
    # Batch size of 1 does not need normalization
    # if batch_norm:
    #   x = BatchNormalization(axis=-1, momentum=0.99,
    #                       epsilon=0.001)(model.output)

    x = Dropout(dropout_prob)(model.output)

    x = conv3d_bn(x, NUM_CLASSES, 1, 1, 1, padding='same',
                  use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, NUM_CLASSES))(x)

    # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
               output_shape=lambda s: (s[0], s[2]))(x)

    # if not endpoint_logit
    x = Activation('softmax', name='prediction')(x)

    # compile the model
    return Model(input=model.input, output=x)


def plot_training_info(case, metrics, save, history):
    """
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png' will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    """
    plt.ioff()
    if 'accuracy' in metrics:
        fig = plt.figure()
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if save:
            plt.savefig(case + 'accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # summarize history for loss
    if 'loss' in metrics:
        fig = plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.legend(['train', 'val'], loc='upper left')
        if save:
            plt.savefig(case + 'loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)


def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in (model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in (model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def miscellaneous(model):
    rgb_sample = np.load(
        r"C:\Users\kentw\OneDrive - University of Toronto\PycharmProjects\kinetics-i3d\data\MAA\idapt518_sub253_DF_10-18-15.npy")
    rgb_sample = np.expand_dims(rgb_sample, axis=0)
    rgb_logits = model.predict(rgb_sample)
    print(rgb_logits.shape)
