"""
The file is simply written to illustrate how a mean roc curve is drawn.
Here I use subject-wise cross validation for example
"""

from __future__ import absolute_import
from scripts.experiment import *
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, roc_curve
import matplotlib
import os
matplotlib.use('TkAgg')


def run_over(data):
    # Import data
    classes = list_data(data, experiments)
    class0, class1 = classes['fail'], classes['pass']

    X_full = np.expand_dims((class0 + class1), axis=1)
    _y_full = np.concatenate(((np.zeros(shape=(len(class0), 1))), np.ones(shape=(len(class1), 1))))
    sub_full = np.asarray(extract_subID((class0 + class1)))

    temp = sorted(zip(X_full, _y_full), key=lambda x: x[0][0][12:15])
    X_full = np.asarray([item[0] for item in temp])
    _y_full = np.asarray([item[1] for item in temp])
    sub_full.sort()
    sub_full = np.unique(sub_full)

    predictions, samples = [], []  # To record all prediction values
    fold_number = 1

    # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
    from sklearn.model_selection import LeaveOneOut, train_test_split
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(sub_full):
        sub_train, sub_test = sub_full[train_index], sub_full[test_index]

        # Find the data matching subject ID
        X_full, _y_full = X_full.flatten(), _y_full.flatten()
        # index_train = [False if sub_test[0] in x else True for x in X_full]
        index_test = [True if sub_test[0] in x else False for x in X_full]
        X2, _y2 = X_full[index_test], _y_full[index_test]

        fold_best_model_path = best_model_path + name + '_MAA_fold_{}.h5'.format(fold_number)
        model = load_model(fold_best_model_path)
        test_generator = DataGenerator(X2, _y2, shuffle=False)
        predicted = model.predict_generator(test_generator)
        y_score = np.copy(predicted)
        predictions.append(y_score)
        samples.append(_y2)
        # break
        fold_number += 1

    return predictions, samples


if __name__ == '__main__':

    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    cross_validation_type = "subject-wise"
    data = "full"
    checkpoint_params = {'batch_norm': False, 'gpu_num': 1, 'learning_rate': 0.01, 'mini_batch_size': 1,
                         'weight_0': 1, 'epochs': 60, 'dropout_prob': 0.36, 'use_checkpoint': True, 'patience': 8, }
    input_dimension = {'NUM_FRAMES': 64, 'FRAME_HEIGHT': 224, 'FRAME_WIDTH': 224, 'NUM_RGB_CHANNELS': 3,
                       'NUM_CLASSES': 2}

    locals().update(checkpoint_params)
    name = r'{}_{}_lr{}_batchs{}_batchnorm{}_w0_{}'.format(cross_validation_type,
                                                           data,
                                                           learning_rate,
                                                           mini_batch_size,
                                                           batch_norm,
                                                           weight_0,
                                                           )
    predictions, samples = run_over(data)

    y_score = np.vstack(predictions)
    samples = np.hstack(samples)

    # LOOCV
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()  # For ROC

    # Binarize the output
    _y2 = label_binarize(samples, classes=[1, 0])
    n_classes = _y2.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    font = {'family': 'Palatino Linotype', 'size': 13}
    plt.rc('font', **font)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(_y2[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    plt.plot(fpr[0], tpr[0],
             lw=lw, color='b',
             label='AUC = %0.2f' % roc_auc[0])

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontfamily='Palatino Linotype', fontsize=13)
    plt.ylabel('True Positive Rate', fontfamily='Palatino Linotype', fontsize=13)
    plt.title('LOSO Experiment ROC Curves of Class Slips', fontfamily='Palatino Linotype')
    plt.legend(loc="lower right")

    # ROC curve plotting
    interp_tpr = np.interp(mean_fpr, fpr[0], tpr[0])
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc[0])

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], )
    ax.legend(loc="lower right", fontsize='medium')

    # plt.savefig('ROC for LOSO Experiment test')
    plt.show()
