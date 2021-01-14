import sys
import argparse
import numpy as np
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut, train_test_split
from numpy.random import seed
from proto.trainer import *

sys.path.append("..")
seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use('Agg')

# Metrics
sensitivities, specificities = [], []
fars, mdrs = [], []
accuracies, f1s = [], []
tps = []
tns = []
fps = []
fns = []


def main(cross_validation_type, data):
    fold_number = 1
    checkpoint_params['fold_number'] = fold_number
    use_checkpoint = checkpoint_params['use_checkpoint']

    # Import data
    classes = list_data(data, experiments)
    class0, class1 = classes['fail'], classes['pass']

    X_full = np.expand_dims((class0 + class1), axis=1)
    _y_full = np.concatenate(((np.zeros(shape=(len(class0), 1))), np.ones(shape=(len(class1), 1))))

    if cross_validation_type == "subject-wise":
        sub_full = np.asarray(extract_subID((class0 + class1)))
        temp = sorted(zip(X_full, _y_full), key=lambda x: x[0][0][12:15])
        X_full = np.asarray([item[0] for item in temp])
        _y_full = np.asarray([item[1] for item in temp])
        sub_full.sort()
        sub_full = np.unique(sub_full)

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
            train(X_train, y_train, X_val, y_val, X2, _y2, checkpoint_params, use_checkpoint)

            reset_keras()
            fold_number += 1
            checkpoint_params['fold_number'] = fold_number

    else:  # Record-wise
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

        # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
        for ((train_index_0, test_index_0), (train_index_1, test_index_1)) in zip(
                kf_0.split(X_full[zeroes_full, ...]), kf_1.split(X_full[ones_full, ...])):
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
            val_size = 0.5

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

            train(X_train, y_train, X_val, y_val, X2, _y2, checkpoint_params, use_checkpoint)

            reset_keras()
            fold_number += 1
            checkpoint_params['fold_number'] = fold_number

    print('CROSS-VALIDATION RESULTS ===================')
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


def train(X_train, y_train, X_val, y_val, X2, _y2, checkpoint_params, use_checkpoint):
    # Generators
    mini_batch_size = checkpoint_params['mini_batch_size']
    gpu_num = checkpoint_params['gpu_num']

    training_generator = DataGenerator(X_train, y_train, mini_batch_size * gpu_num)
    validation_generator = DataGenerator(X_val, y_val, mini_batch_size * gpu_num)

    input_dimension = {'NUM_FRAMES': 64, 'FRAME_HEIGHT': 224, 'FRAME_WIDTH': 224, 'NUM_RGB_CHANNELS': 3,
                       'NUM_CLASSES': 2}
    trainer = Trainer(checkpoint_params,
                      training_generator,
                      validation_generator,
                      input_dimension)
    if not use_checkpoint:
        trainer.train(save_plots=False)
    return evaluate(X2, _y2, trainer)


def evaluate(X2, _y2, trainer):
    test_generator = DataGenerator(X2, _y2, shuffle=False)
    predicted, y_score = trainer.evaluate(test_generator)
    metrics = trainer.compute_metrics(predicted, _y2)
    aggregate_statistics(metrics)
    return predicted, y_score


def aggregate_statistics(metrics):
    tp = metrics['tp']
    fn = metrics['fn']
    tn = metrics['tn']
    fp = metrics['fp']
    fpr = metrics['fpr']
    fnr = metrics['fnr']
    accuracy = metrics['accuracy']
    f1 = metrics['f1']

    # Store the metrics for this fold
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


def get_args(argv=None):

    parser = argparse.ArgumentParser(description="detect fall events in MAA footwear testing")
    parser.add_argument('--cross_validation_type', type=str,
                        help='type of cross validation')
    parser.add_argument('--data', type=str, default="full",
                        help='volume of data used')

    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')

    parser.add_argument('--use_checkpoint', type=str_to_bool, default=True,
                        help='if checkpoint is used')
    parser.add_argument('--batch_norm', type=str_to_bool, default=False,
                        help='batch normalization')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='the number of GPUs (activate more cuda visible devices)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--mini_batch_size', type=int, default=1,
                        help='the size of mini batch')
    parser.add_argument('--weight_0', type=float, default=1,
                        help='the weight of the negative class in binary classification')
    parser.add_argument('--epochs', type=int, default=60,
                        help='the number of epochs')
    parser.add_argument('--dropout_prob', type=float, default=0.36,
                        help='dropout layer probability')
    parser.add_argument('--patience', type=int, default=60,
                        help='patience for early stopping')
    parser.add_argument('-l', '--input_dimension', nargs='+', default=[64, 224, 224, 3, 2],
                        help='NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS and NUM_CLASSES')
    return parser.parse_args(argv)


if __name__ == '__main__':
    parser = get_args()

    cross_validation_type = parser.cross_validation_type
    data = parser.data

    checkpoint_params = {'batch_norm': parser.batch_norm, 'gpu_num': parser.gpu_num,
                         'learning_rate': parser.learning_rate, 'mini_batch_size': parser.mini_batch_size,
                         'weight_0': parser.weight_0, 'epochs': parser.epochs,
                         'dropout_prob': parser.dropout_prob, 'use_checkpoint': parser.use_checkpoint,
                         'patience': parser.patience}
    input_dimension = {'NUM_FRAMES': parser.input_dimension[0],
                       'FRAME_HEIGHT': parser.input_dimension[1],
                       'FRAME_WIDTH': parser.input_dimension[2],
                       'NUM_RGB_CHANNELS': parser.input_dimension[3],
                       'NUM_CLASSES': parser.input_dimension[4]}

    locals().update(checkpoint_params)
    name = r'{}_{}_lr{}_batchs{}_batchnorm{}_w0_{}'.format(cross_validation_type,
                                                           data,
                                                           learning_rate,
                                                           mini_batch_size,
                                                           batch_norm,
                                                           weight_0,
                                                           )
    checkpoint_params['name'] = name
    main(cross_validation_type, data)
