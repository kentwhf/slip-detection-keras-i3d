#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from __future__ import print_function
from numpy.random import seed

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras as keras

from DataGenerator import DataGenerator
from i3d_inception import *
from utils import *

# Helper libraries
seed(1)
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, ZeroPadding2D)
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from scipy import interp
from keras.layers.advanced_activations import ELU

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


# In[2]:


# CHANGE THESE VARIABLES ---
save_plots = True
use_checkpoint = False # Set to True or False

# Paths
best_model_path = 'test/'
plots_folder = 'plots/'

# Hyper parameters
batch_norm = False
gpu_num = 1 
learning_rate = 0.01
mini_batch_size = 1
weight_0 = 1 # higher weight of 0 prioritizes learning in class 0
epochs = 60
dropout_prob=0.36

# Input dimensions
NUM_FRAMES = 64
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_CLASSES = 2

optimizer = 'adam'
weights = "rgb_imagenet_and_kinetics"
name = 'LOSO'

# Name of the experiment
exp = '_lr{}_batchs{}_batchnorm{}_w0_{}_{}_{}'.format(learning_rate,
                                                      mini_batch_size,
                                                      batch_norm,
                                                      weight_0, 
                                                      name,
                                                      weights)


# In[3]:


# trainer 
def main(exp_name, experiments):
    global exp
    exp = exp_name + exp
    
    compute_metrics = True
    threshold = 0.5

    # Import data
    classes = list_data(exp_name, experiments)
    class0, class1 = classes['fail'], classes['pass']

    X_full = np.expand_dims((class0 + class1), axis=1)
    _y_full = np.concatenate(((np.zeros(shape = (len(class0),1))), np.ones(shape = (len(class1),1))))
    sub_full = np.asarray(extract_subID((class0 + class1)))
    
    temp = sorted(zip(X_full, _y_full), key=lambda x: x[0][0][12:15])
    X_full = np.asarray([item[0] for item in temp])
    _y_full = np.asarray([item[1] for item in temp])
    sub_full.sort()
    sub_full = np.unique(sub_full)

    sensitivities = []
    specificities = []
    fars = []
    mdrs = []
    accuracies = []
    
    f1s = []
    tps = []
    tns = []
    fps = []
    fns = []
    tprs = []
    aucs = []    
    
    fold_number = 1 

#     mean_fpr = np.linspace(0, 1, 100)
#     fig, ax = plt.subplots() # For ROC
    
    # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
    from sklearn.model_selection import LeaveOneOut, train_test_split
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(sub_full):
        sub_train, sub_test = sub_full[train_index], sub_full[test_index]

        # Find the data matching subject ID
        X_full, _y_full = X_full.flatten(), _y_full.flatten()
        index_train = [False if sub_test[0] in x else True for x in X_full]
        index_test = [True if sub_test[0] in x else False for x in X_full]
        X_train, y_train = X_full[index_train], _y_full[index_train]
        X2, _y2 = X_full[index_test], _y_full[index_test]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                          test_size=0.1, random_state=42)

        # Generators
        training_generator = DataGenerator(X_train, y_train, mini_batch_size*gpu_num)
        validation_generator = DataGenerator(X_val, y_val, mini_batch_size*gpu_num)

        reset_keras()
        model = build_model()
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt = SGD(lr=learning_rate, momentum=0.0, nesterov=False)
        model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])

        fold_best_model_path = best_model_path + exp + '_MAA_fold_{}.h5'.format(fold_number)

        if not use_checkpoint:
            # ==================== TRAINING ========================
            # weighting of each class: only the fall class gets a different weight
            class_weight = {0: weight_0, 1: 1}

            # callback definition
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
            history = model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          steps_per_epoch=len(training_generator),
                                          nb_epoch=epochs,
                                          class_weight=class_weight,
                                          callbacks=callbacks,
                                          use_multiprocessing=True,
                                          workers=6,
                                          shuffle=True)

            plot_training_info(plots_folder + exp + '_fold{}'.format(fold_number), ['accuracy', 'loss'],
                               save_plots, history.history)

        # ==================== EVALUATION ========================

        # Load best model
        print('\nModel loaded from checkpoint')
        print(sub_test[0])

        model = load_model(fold_best_model_path)
        test_generator = DataGenerator(X2, _y2, shuffle=False)
        
        if compute_metrics:
            predicted = model.predict_generator(test_generator)
            y_score = np.copy(predicted)

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

            # Store the metrics for this epoch
            sensitivities.append(tp / float(tp + fn))
            specificities.append(tn / float(tn + fp))
            fars.append(fpr)
            mdrs.append(fnr)
            accuracies.append(accuracy)
            f1s.append(f1)

            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)

#             # Binarize the output
#             _y2 = label_binarize(_y2, classes=[1, 0])
#             n_classes = _y2.shape[1] 

#             fpr = dict()
#             tpr = dict()
#             roc_auc = dict()

#             for i in range(n_classes):
#                 fpr[i], tpr[i], _ = roc_curve(_y2[:, i], y_score[:, i])
#                 roc_auc[i] = auc(fpr[i], tpr[i])

#             lw = 2
#             color = ['g', 'r', 'c', 'm', 'y']
#             plt.plot(fpr[0], tpr[0],
#                      lw=lw, color=color[fold_number - 1],
#                      label= 'ROC fold {}'.format(fold_number) + ' ' + '(area = %0.2f)' % roc_auc[0])
#             plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate', fontfamily= 'Palatino Linotype')
#             plt.ylabel('True Positive Rate', fontfamily= 'Palatino Linotype')
#             plt.title('Experiment 4 ROC Curves of Class Slips', fontfamily= 'Palatino Linotype')
#             plt.legend(loc="lower right")

# #                 font = {'family' : 'Palatino Linotype', 'size':  10}
# #                 plt.rc('font', **font)

#             # ROC curve plotting
#             interp_tpr = np.interp(mean_fpr, fpr[0], tpr[0])
#             interp_tpr[0] = 0.0
#             tprs.append(interp_tpr)
#             aucs.append(roc_auc[0])

        fold_number += 1

    print('\n18-FOLD CROSS-VALIDATION RESULTS ===================')
    print("Sensitivity: %.2f (+/- %.2f)" % (np.mean(sensitivities), np.std(sensitivities)))
    print("Specificity: %.2f (+/- %.2f)" % (np.mean(specificities), np.std(specificities)))
    print("FAR: %.2f (+/- %.2f)" % (np.mean(fars), np.std(fars)))  # False alarm rates 
    print("MDR: %.2f (+/- %.2f)" % (np.mean(mdrs), np.std(mdrs)))  # Missed detection rates
    print("Accuracy: %.2f (+/- %.2f)" % (np.mean(accuracies), np.std(accuracies)))
    print("F1: %.2f (+/- %.2f)" % (np.mean(f1s), np.std(f1s)))

    print("tp: %.2f (+/- %.2f)" % (np.mean(tps), np.std(tps)))
    print("fp: %.2f (+/- %.2f)" % (np.mean(fps), np.std(fps)))
    print("tn: %.2f (+/- %.2f)" % (np.mean(tns), np.std(tns)))
    print("fn: %.2f (+/- %.2f)" % (np.mean(fns), np.std(fns)))

#     # ROC
#     font = {'family' : 'Palatino Linotype', 'size':  10}
#     plt.rc('font', **font)

#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     ax.plot(mean_fpr, mean_tpr, color='b',
#             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#             lw=2, alpha=.8)

#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')

#     ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],)
#     ax.legend(loc="lower right")

#     plt.show()
#     plt.savefig('ROC for exp4')


# In[ ]:


if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
    # Experiments
    # 1. Only with hazardous slips (with more data you gathered) --> 60:60--> same as last time with more data
    # 2. Only with normal slips and 'no slips' --> 60:60 --> this can be a fair detection 
    # 3. Combination of 1+2 --> 120:120
    # 4. With small slips --> 60:60
    # 5. With all dataset --> 180:180
    
    experiments = {'exp1':['Hazardous_slips', 'Pass_split1'],
                   'exp2':['Normal_slips', 'Pass_split1'],
                   'exp3':['Hazardous_slips', 'Normal_slips', 'Pass_split1', 'Pass_split2'],
                   'exp4':['Small_slips', 'Pass_split1'],
                   'exp5':['Hazardous_slips', 'Normal_slips', 'Pass_split1', 'Pass_split2', 'Small_slips', 'Pass_split3']
                  }
    
    main('exp4', experiments) # Make sure you change ROC manually
    
    # Learning rate
    # Memory usage
    # Patience
    # ROC title and name
    # Not use check model
    # main function argument


# In[ ]:





# 
