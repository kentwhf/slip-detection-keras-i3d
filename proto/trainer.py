from numpy.random import seed
import os
ROOT_DIR = os.path.abspath(os.curdir)

import tensorflow as tf
import tensorflow.keras as keras
from proto.DataGenerator import DataGenerator
from proto.i3d_inception import *
from utils.utils import *

# Helper libraries
from matplotlib import pyplot as plt
import os
import keras.backend as K
import numpy as np
import matplotlib

from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, ZeroPadding2D)
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from scipy import interp

seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use('Agg')

# # Paths
import os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
best_model_path = ROOT_DIR + '/temp1/'
plots_folder = ROOT_DIR + '/temp2/'


class Trainer:
    def __init__(self,
                 checkpoint_params,
                 training_generator,
                 validation_generator,
                 input_dimension,
                 model=None,
                 weights=None,
                 optimizer=None
                 ):
        """Train a DNN using given preprocessing, weights, and data
            
        The purpose of the Trainer is handle a default training mechanism.
        As required input it expects a `dataset` and hyperparameters (`checkpoint_params`).
        The steps are
            1. Loading and preprocessing of the dataset
            2. Computation of the codec
            3. Construct the desired Deep Learning Framework
            4. Launch of the training

        During the training the Trainer will perform validation checks if a `validation_dataset` is given
        to determine the best model. Furthermore, the current status is printet and checkpoints are written.

        Parameters
        ----------
        checkpoint_params : CheckpointParams
            The dictionary object that defines all hyperparameters of the model
        input_dimension: Input Dimension
            The input dimension of objects being trained with
        dataset : Dataset
            The Dataset used for training
        model: Model 
            The model framework defined by user
        validation_dataset : Dataset, optional
            The Dataset used for validation, i.e. choosing the best model
        # save_weight: boolean, optional
        #     If the trained weights are preserved
        weights : str, optional
            Path to a trained model for loading its weights
        optimizer: Optimizer 
            The optimizer used to train throughout the process
        """
        self.checkpoint_params = checkpoint_params
        self.input_dimension = input_dimension
        self.dataset = training_generator
        self.validation_dataset = validation_generator
        self.weights = weights

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        if model:
            self.model = model
        else:
            NUM_FRAMES = input_dimension['NUM_FRAMES']
            FRAME_HEIGHT = input_dimension['FRAME_HEIGHT']
            FRAME_WIDTH = input_dimension['FRAME_WIDTH']
            NUM_RGB_CHANNELS = input_dimension['NUM_RGB_CHANNELS']
            NUM_CLASSES = input_dimension['NUM_CLASSES']
            dropout_prob = checkpoint_params['dropout_prob']

            self.build_model(NUM_FRAMES,
                             FRAME_HEIGHT,
                             FRAME_WIDTH,
                             NUM_RGB_CHANNELS,
                             NUM_CLASSES,
                             dropout_prob,
                             )

        # if len(self.dataset) == 0:
        #     raise Exception("Dataset is empty.")
        #
        # if self.validation_dataset and len(self.validation_dataset) == 0:
        #     raise Exception("Validation dataset is empty. Provide validation data for early stopping.")

    def train(self, save_plots):
        weight_0 = self.checkpoint_params['weight_0']
        epochs = self.checkpoint_params['epochs']
        fold_number = self.checkpoint_params['fold_number']
        name = self.checkpoint_params['name']

        fold_best_model_path = best_model_path + name + '_MAA_fold_{}.h5'.format(fold_number)

        # Training
        # Weighting of each class: only the fall class gets a different weight
        class_weight = {0: weight_0, 1: 1}

        # Callback definition
        # Choose the monitored metric
        metric = 'val_loss'
        e = EarlyStopping(monitor=metric, min_delta=0, patience=8,
                          mode='auto', restore_best_weights=True)
        c = ModelCheckpoint(fold_best_model_path,
                            monitor=metric,
                            save_best_only=True,
                            save_weights_only=False, mode='auto')
        callbacks = [e, c]

        # Batch training
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)

        history = self.model.fit_generator(generator=self.dataset,
                                           validation_data=self.validation_dataset,
                                           steps_per_epoch=len(self.validation_dataset),
                                           nb_epoch=epochs,
                                           class_weight=class_weight,
                                           callbacks=callbacks,
                                           use_multiprocessing=True,
                                           workers=6,
                                           shuffle=True)

        plot_training_info(plots_folder + name + '_fold{}'.format(fold_number), ['accuracy', 'loss'],
                           save_plots, history.history)

    def evaluate(self, test_generator, threshold=0.5):
        # Evaluate
        # Load best model
        # weight = self.checkpoint_params['weight']
        print('\nModel loaded from checkpoint')

        fold_number = self.checkpoint_params['fold_number']
        name = self.checkpoint_params['name']

        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
        fold_best_model_path = best_model_path + name + '_MAA_fold_{}.h5'.format(fold_number)

        self.model = load_model(fold_best_model_path)
        predicted = self.model.predict_generator(test_generator)
        y_score = np.copy(predicted)

        predicted = predicted[:, 1]

        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1

        # Binary array of predictions 
        predicted = np.asarray(predicted).astype(int)
        return predicted, y_score

    def compute_metrics(self, predicted, ground_true):
        fold_number = self.checkpoint_params['fold_number']

        # Compute metrics and print them
        cm = confusion_matrix(ground_true, predicted, labels=[0, 1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]

        tpr = tp / float(tp + fn)
        fpr = fp / float(fp + tn)
        fnr = fn / float(fn + tp)
        tnr = tn / float(tn + fp)
        accuracy = accuracy_score(ground_true, predicted)
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
        return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
                'fpr': fpr, 'fnr': fnr, 'accuracy': accuracy, 'f1': f1}


    def build_model(self,
                    NUM_FRAMES,
                    FRAME_HEIGHT,
                    FRAME_WIDTH,
                    NUM_RGB_CHANNELS,
                    NUM_CLASSES,
                    dropout_prob,
                    weights="rgb_imagenet_and_kinetics"
                    ):
        # I3D ARCHITECTURE
        model = Inception_Inflated3d(
            include_top=False,
            weights=weights,
            input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
            classes=NUM_CLASSES)

        # ==================== CLASSIFIER ========================
        # Batch size of 1 does not need normalization
        batch_norm = self.checkpoint_params['batch_norm']
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99,
                                   epsilon=0.001)(model.output)
            x = Dropout(dropout_prob)(x)
        else:
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
        self.model = Model(input=model.input, output=x)
        self.model.compile(self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])


if __name__ == '__main__':
    checkpoint_params = {'batch_norm': False, 'gpu_num': 1, 'learning_rate': 0.01, 'mini_batch_size': 1,
                         'weight_0': 1, 'epochs': 60, 'dropout_prob': 0.36}
    input_dimension = {'NUM_FRAMES': 64, 'FRAME_HEIGHT': 224, 'FRAME_WIDTH': 224, 'NUM_RGB_CHANNELS': 3,
                       'NUM_CLASSES': 2}
    # name = self.checkpoint_params['name']
    # weights = self.checkpoint_params['weights']
    # path = self.checkpoint_params['path']
    # fold_number = self.checkpoint_params['fold_number']

    reset_keras()
    # opt = SGD(lr=learning_rate, momentum=0.0, nesterov=False)

    trainer = Trainer(checkpoint_params,
                      None,
                      None,
                      input_dimension)


