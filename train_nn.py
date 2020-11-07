#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from __future__ import print_function
from numpy.random import seed

# TensorFlow and tf.keras
import tensorflow as tf
import keras

from DataGenerator import DataGenerator
from i3d_inception import *

# Helper libraries
seed(1)
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import h5py
import scipy.io as sio
import glob
import gc

from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, ZeroPadding2D)
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.layers.advanced_activations import ELU

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend.tensorflow_backend as K

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session


# In[2]:


# CHANGE THESE VARIABLES ---
data_folder = r'C:\Users\kentw\OneDrive - University of Toronto\PycharmProjects\kinetics-i3d\data\MAA' 
#mean_file = 'flow_mean.mat'  # Used as a normalisation for the input of the network

save_features = False  # Boolean flag if we save features in h5py
save_plots = True
load_data = False

# Set to 'True' if you want to restore a previous trained models
# Training is skipped and test is done
use_checkpoint = False # Set to True or False
# --------------------------

best_model_path = 'models/'
plots_folder = 'plots/'
checkpoint_path = 'models/fold_'

features_file = 'features_MAA.h5'
labels_file = 'labels_MAA.h5'
features_key = 'features'
labels_key = 'labels'

rgb_file = 'rgb_MAA.h5'
rgb_key = 'rgb'

# Hyper parameters
batch_norm = False
gpu_num = 1 
num_features = [None, 7, 1, 1, 1024]  # Specific dimension of features for 64-frame clips
learning_rate = 0.1
mini_batch_size = 1
weight_0 = 1.0  # A higher weight of 0 prioritizes learning in class 0
epochs = 1
dropout_prob=0.0

weights = 'rgb_imagenet_and_kinetics'
name = 'train_nn'

# Name of the experiment
exp = 'slips_lr{}_batchs{}_batchnorm{}_w0_{}_{}_{}'.format(learning_rate,
                                                           mini_batch_size,
                                                           batch_norm,
                                                           weight_0, 
                                                           name,
                                                           weights)
# Input dimensions
NUM_FRAMES = 64
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_CLASSES = 2


# In[3]:


# Functions 
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
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
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


def list_data():
    """
    Function to list existing data set by filename and label 
    """
    vids = {'pass': [], 'fail': []}
    
    for file in os.listdir(data_folder):
        if not file.lower().endswith('.npy'):
            continue
        else:
            file_dir = os.path.join(data_folder, file)
            ID = file_dir.split('\\')[-1].rstrip('.npy')
            # Classify
            result = ID.split('_')[2][1]
            if result == "P": vids["pass"].append(ID)
            else: vids["fail"].append(ID)
    
    return vids


def saveFeatures(feature_extractor,
                 features_file,
                 labels_file,
                 features_key,
                 labels_key):
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
    classes = list_data()
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
        dataset_labels[i, :]= 1

        del rgb_images, rgb_labels
        gc.collect()

    h5features.close()
    h5labels.close()
    
    
# def load_data(rgb_file,
#               labels_file,
#               rgb_key,
#               labels_key):
#     """
#     Function to load the existing data set into h5py format 
#     """
#     # Fill the IDs and classes arrays
#     classes = list_data()
#     class0, class1 = classes['fail'], classes['pass']
#     num_samples = len(class0) + len(class1)
#     num_features[0] = num_samples
    
#     # File to store the extracted features and datasets to store them
#     # IMPORTANT NOTE: 'w' mode totally erases previous data
#     # Write files in h5py format
#     h5rgb = h5py.File(rgb_file, 'w')
#     h5labels = h5py.File(labels_file, 'w')
    
#     # Create data sets
#     dataset_rgb = h5rgb.create_dataset(rgb_key,
#                                             shape=(num_samples, 64, 224, 224, 3),
#                                             dtype='float64')
#     dataset_labels = h5labels.create_dataset(labels_key,
#                                              shape=(num_samples, 1),
#                                              dtype='float64')

#     # Process class 0 
#     gen = DataGenerator(class0, np.zeros(len(class0)), 1)
#     for i in range(len(class0)):
                        
#         rgb_images, rgb_labels = gen.__getitem__(i)

#         dataset_rgb[i, :] = rgb_images
#         dataset_labels[i, :] = 0
                
#         del rgb_images, rgb_labels
#         gc.collect()
        
#     # Process class 1
#     gen = DataGenerator(class1, np.ones(len(class1)), 1)
#     for i in range(len(class0), num_samples):
        
#         rgb_images, rgb_labels = gen.__getitem__(i)        
        
#         dataset_rgb[i, :] = rgb_images
#         dataset_labels[i, :]= 1

#         del rgb_images, rgb_labels
#         gc.collect()

#     h5rgb.close()
#     h5labels.close()


# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))


# In[4]:


# trainer 
def main():
    # ========================================================================
    # I3D ARCHITECTURE
    # ========================================================================
    base_model = Inception_Inflated3d(
        include_top=False ,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
        classes=NUM_CLASSES)

    # ========================================================================
    # TRAINING
    # ========================================================================    
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08)
    # base_model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  # metrics=['accuracy'])
    
    do_training = True
    compute_metrics = True
    threshold = 0.5

    if do_training:
        # Import data
        if load_data:
            load_data(rgb_file,
                      labels_file,
                      rgb_key,
                      labels_key)
            
        # # Import data
        # h5rgb = h5py.File(rgb_file, 'r')
        # h5labels = h5py.File(labels_file, 'r')
        
        classes = list_data()
        class0, class1 = classes['fail'], classes['pass']
        X_full = np.asarray((class0 + class1))
        X_full = np.expand_dims(X_full, axis=1)
        _y_full = np.concatenate(((np.zeros(shape = (len(class0),1))), np.ones(shape = (len(class1),1))))
        
        # X_full will contain all the feature vectors extracted
        # X_full = h5rgb[rgb_key]
        # _y_full = np.asarray(h5labels[labels_key])
        # print(X_full, _y_full)
        
        # Indices of 0 and 1 in the data set
        zeroes_full = np.asarray(np.where(_y_full == 0)[0])
        ones_full = np.asarray(np.where(_y_full == 1)[0])
        zeroes_full.sort()
        ones_full.sort()

        # Traditional Machine Learning methodology
        # Method get_n_splits() returns the number of splits
        
        kf_0 = KFold(n_splits=5, shuffle=True)
        kf_0.get_n_splits(X_full[zeroes_full, ...])

        kf_1 = KFold(n_splits=5, shuffle=True)
        kf_1.get_n_splits(X_full[ones_full, ...])

        sensitivities = []
        specificities = []
        fars = []
        mdrs = []
        accuracies = []

        fold_number = 1
        
        # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
        for ((train_index_0, test_index_0),
             (train_index_1, test_index_1)) in zip(
            kf_0.split(X_full[zeroes_full, ...]),
            kf_1.split(X_full[ones_full, ...])
        ):
            
            train_index_0 = np.asarray(train_index_0)
            test_index_0 = np.asarray(test_index_0)
            train_index_1 = np.asarray(train_index_1)
            test_index_1 = np.asarray(test_index_1)

            # Train and Test Set
            X = np.concatenate((X_full[zeroes_full, ...][train_index_0, ...],
                                X_full[ones_full, ...][train_index_1, ...]))
            _y = np.concatenate((_y_full[zeroes_full, ...][train_index_0, ...],
                                 _y_full[ones_full, ...][train_index_1, ...]))
            X2 = np.concatenate((X_full[zeroes_full, ...][test_index_0, ...],
                                 X_full[ones_full, ...][test_index_1, ...]))
            _y2 = np.concatenate((_y_full[zeroes_full, ...][test_index_0, ...],
                                  _y_full[ones_full, ...][test_index_1, ...]))

            # Create a validation subset from the training set
            val_size = 0.2
            
            zeroes = np.asarray(np.where(_y == 0)[0])
            ones = np.asarray(np.where(_y == 1)[0])
            
            zeroes.sort()
            ones.sort()

            trainval_split_0 = StratifiedShuffleSplit(n_splits=1,
                                                      test_size=val_size / 2,
                                                      random_state=1)
            indices_0 = trainval_split_0.split(X[zeroes, ...],
                                               np.argmax(_y[zeroes, ...], 1))
            trainval_split_1 = StratifiedShuffleSplit(n_splits=1,
                                                      test_size=val_size / 2,
                                                      random_state=1)
            indices_1 = trainval_split_1.split(X[ones, ...],
                                               np.argmax(_y[ones, ...], 1))
            
            train_indices_0, val_indices_0 = indices_0.__next__()
            train_indices_1, val_indices_1 = indices_1.__next__()

            X_train = np.concatenate([X[zeroes, ...][train_indices_0, ...],
                                      X[ones, ...][train_indices_1, ...]], axis=0)
            y_train = np.concatenate([_y[zeroes, ...][train_indices_0, ...],
                                      _y[ones, ...][train_indices_1, ...]], axis=0)
            X_val = np.concatenate([X[zeroes, ...][val_indices_0, ...],
                                    X[ones, ...][val_indices_1, ...]], axis=0)
            y_val = np.concatenate([_y[zeroes, ...][val_indices_0, ...],
                                    _y[ones, ...][val_indices_1, ...]], axis=0)
            
            X_train = X_train.flatten()
            X_val = X_val.flatten()
            y_train = y_train.flatten()
            y_val = y_val.flatten()
            
            X2 = X2.flatten()
            _y2 = _y2.flatten()
            
            
            # Generators
            training_generator = DataGenerator(X_train, y_train)
            validation_generator = DataGenerator(X_val, y_val)
            
            # ==================== CLASSIFIER ========================

            ## Batch size of 1 does not need normalization
            # if batch_norm:
            #    x = BatchNormalization(axis=-1, momentum=0.99,
            #                           epsilon=0.001)(extracted_features)

            # Classification block
            x = Dropout(dropout_prob)(base_model.output)

            x = conv3d_bn(x, NUM_CLASSES, 1, 1, 1, padding='same', 
                    use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

            num_frames_remaining = int(x.shape[1])
            x = Reshape((num_frames_remaining, NUM_CLASSES))(x)

            # logits (raw scores for each class)
            x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                       output_shape=lambda s: (s[0], s[2]))(x)

            # if not endpoint_logit
            x = Activation('softmax', name='prediction')(x)

            #compile the model
            model = Model(input = base_model.input, output = x)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            fold_best_model_path = best_model_path + exp + '_MAA_fold_{}.h5'.format(fold_number)
            
            if not use_checkpoint:
                # ==================== TRAINING ========================
                # weighting of each class: only the fall class gets
                # a different weight
                class_weight = {0: weight_0, 1: 1}

                # callback definition
                metric = 'val_loss'
                e = EarlyStopping(monitor=metric, min_delta=0, patience=10,
                                  mode='auto')
                c = ModelCheckpoint(fold_best_model_path, monitor=metric,
                                    save_best_only=True,
                                    save_weights_only=False, mode='auto')
                callbacks = [e, c]

                # validation_generator.len = 2
                
                # Batch training
                if mini_batch_size == 0:
                    history = model.fit_generator(generator=training_generator,
                                             validation_data=validation_generator,
                                             steps_per_epoch=len(training_generator),
                                             nb_epoch=epochs,
                                             class_weight=class_weight,
                                             callbacks=callbacks, 
                                             use_multiprocessing=True,
                                             workers=6)
                else:
                    history = model.fit_generator(generator=training_generator,
                                             validation_data=validation_generator,
                                             steps_per_epoch=len(training_generator),
                                             nb_epoch=epochs,
                                             class_weight=class_weight,
                                             callbacks=callbacks, 
                                             use_multiprocessing=True,
                                             workers=6)

                plot_training_info(plots_folder + exp, ['accuracy', 'loss'],
                                   save_plots, history.history)

                model = load_model(fold_best_model_path)

                # Use full training set (training+validation)
                # Could learn from training set only 
                # Not sure why it should be saved 
                
                reset_keras()
                
                X_train = np.concatenate((X_train, X_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

                training_generator = DataGenerator(X_train, y_train)
                
                if mini_batch_size == 0:
                    history = model.fit_generator(generator=training_generator,
                                             steps_per_epoch=len(training_generator),
                                             nb_epoch=1,
                                             class_weight=class_weight,
                                             callbacks=callbacks, 
                                             use_multiprocessing=True,
                                             workers=6)
                else:
                    history = model.fit_generator(generator=training_generator,
                                             steps_per_epoch=len(training_generator),
                                             nb_epoch=1,
                                             class_weight=class_weight,
                                             callbacks=callbacks, 
                                             use_multiprocessing=True,
                                             workers=6)

                model.save(fold_best_model_path)

                reset_keras()
                
            # ==================== EVALUATION ========================
            # Load best model
            print('Model loaded from checkpoint')

            model = load_model(fold_best_model_path)
            
            training_generator = DataGenerator(X2, _y2)
            
            if compute_metrics:
                predicted = model.predict_generator(training_generator)
                predicted = predicted[:,1]
                for i in range(len(predicted)):
                    if predicted[i] < threshold:
                        predicted[i] = 0
                    else:
                        predicted[i] = 1

                # Array of predictions 0/1
                predicted = np.asarray(predicted).astype(int)
                
                # Compute metrics and print them
                cm = confusion_matrix(_y2, predicted, labels=[0, 1])
                tp = cm[0][0]
                fn = cm[0][1]
                fp = cm[1][0]
                tn = cm[1][1]
                
                tpr = tp / float(tp + fn)
                fpr = fp / float(fp + tn)
                fnr = fn / float(fn + tp)
                tnr = tn / float(tn + fp)
                accuracy = accuracy_score(_y2, predicted)
                precision = tp / float(tp + fp)
                recall = tp / float(tp + fn)
                specificity = tn / float(tn + fp)
        
                try:
                    f1 = 2 * float(precision * recall) / float(precision + recall)                
                except:
                    f1 = 0
                    print("An exception occurred")
                
                print('\n')
                print('FOLD {} results:'.format(fold_number))
                print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))
                print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr, tnr, fpr, fnr))
                print('Sensitivity/Recall: {}'.format(recall))
                print('Specificity: {}'.format(specificity))
                print('Precision: {}'.format(precision))
                print('F1-measure: {}'.format(f1))
                print('Accuracy: {}'.format(accuracy))
                print('\n')

                fold_number += 1

                # Store the metrics for this epoch
                sensitivities.append(tp / float(tp + fn))
                specificities.append(tn / float(tn + fp))
                fars.append(fpr)
                mdrs.append(fnr)
                accuracies.append(accuracy)
                
                reset_keras()

    print('5-FOLD CROSS-VALIDATION RESULTS ===================')
    print("Sensitivity: %.2f (+/- %.2f)" % (np.mean(sensitivities), np.std(sensitivities)))
    print("Specificity: %.2f (+/- %.2f)" % (np.mean(specificities), np.std(specificities)))
    print("FAR: %.2f (+/- %.2f)" % (np.mean(fars), np.std(fars)))  # False alarm rates 
    print("MDR: %.2f (+/- %.2f)" % (np.mean(mdrs), np.std(mdrs)))  # Missed detection rates
    print("Accuracy: %.2f (+/- %.2f)" % (np.mean(accuracies), np.std(accuracies)))


# In[5]:


if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    main()


# In[ ]:


# Miscellaneous
def miscellaneous():
    model = Inception_Inflated3d(
            include_top=False,
            weights='rgb_imagenet_and_kinetics',
            input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
            classes=NUM_CLASSES)

    rgb_sample = np.load(r"C:\Users\kentw\OneDrive - University of Toronto\PycharmProjects\kinetics-i3d\data\MAA\idapt518_sub253_DF_10-18-15.npy")
    rgb_sample = np.expand_dims(rgb_sample, axis=0)
    rgb_logits = model.predict(rgb_sample)
    print(rgb_logits.shape)
    
    
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

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
    
    
# miscellaneous()

model = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
        classes=NUM_CLASSES)
get_model_memory_usage(1, model)


# In[ ]:


from datetime import datetime

# Define global variables
root_dir = "C:/Users/kentw/OneDrive - University of Toronto/PycharmProjects/Fall-Detection-with-CNNs-and-Optical-Flow/"
video_folder = os.path.join(root_dir, "raw_data")  # Folder of MP4 Data
label_folder = os.path.join(root_dir, "data_annotation")

dates = [f for f in os.listdir(video_folder)]
video_paths, label_paths = [os.path.join(video_folder, d) for d in dates], [os.path.join(label_folder, d) for d in dates]

# Make a list of labels sort by date
# label/video paths differentiable by dates
# i-th folder denotes a certain date
for i in range(len(label_paths)):
    labels = glob.glob(label_paths[i] + "/*.mat")
    videos = glob.glob(video_paths[i] + "/*.mp4")
    labels = sorted(labels, key=lambda x: datetime.strptime(x[-12:-4], "%H-%M-%S"))
    
    print('\nFinished one-to-one mapping for {}'.format(dates[i]))
    
    for x,y in zip(videos, labels):
        print(x)
        print(y)
        print('\n')




