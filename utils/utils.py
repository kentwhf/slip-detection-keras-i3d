import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
from keras.backend.tensorflow_backend import clear_session, get_session
import keras.backend as K
matplotlib.use('Agg')

experiments = {'maxi': ['Hazardous_slips', 'Pass_split1'],
               'midi': ['Normal_slips', 'Pass_split1'],
               'fair detection': ['Hazardous_slips', 'Normal_slips', 'Pass_split1', 'Pass_split2'],
               'mini': ['Small_slips', 'Pass_split1'],
               'full': ['Hazardous_slips', 'Normal_slips', 'Pass_split1', 'Pass_split2', 'Small_slips', 'Pass_split3']
               }


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
    """
    Function to extract subject IDs
    """
    res = []
    for item in list_:
        sub = item.split('_')[1]
        res.append(sub)
    return res


def reset_keras():
    """
    Function to reset Keras sessions to avoid overload
    """
    sess = get_session()
    clear_session()
    sess.close()

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))



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
    """
    Function to compute the amount of memory used
    """
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
