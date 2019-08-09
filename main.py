#!/usr/bin/env python3

'''
# How to install dependencies on Arch Linux
sudo pip install -U scikit-learn
sudo pip install -U matplotlib
sudo pip install -U pandas
sudo pacman -S tk
'''

import warnings

warnings.filterwarnings("ignore")

import argparse
import os.path
import collections
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams.update({'font.size': 8})

__default_ts_name = 'breast-cancer'
__default_test_size = 0.20
__default_question_mark_count = 4
__default_question_mark_count_repeated = 1
__default_no_split = False
__default_train_test_split_random_state = 300

__default_knn_k = 6
__deafault_random_forest_estimator = 100
__deafult_model_classifier = ('MLP', 'SVM', 'RandomForest', 'KNN')

__default_training_verbosity = 0
__default_imgs_path = './imgs/'
__default_csv_path = './csv/'
__default_save_img = True

__ts_opts = {
    'breast-cancer': {
        'url': "./ts/breast-cancer-wisconsin.data",
        'columns': ('Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli',
                    'Mitoses', 'Class'),
        'classes_name': ['benign', 'malignant'],
        'question_mark_count': __default_question_mark_count,
        'x_slice': (slice(None, None), slice(1, -1)),
        'y_slice': (slice(None, None), -1),
        'mlp_hidden_layers_sizes': [180, 60, 20],
        'knn_k': __default_knn_k,
        'random_forest_estimator': __deafault_random_forest_estimator,
    },
    'letters': {
        'url': "./ts/letter-recognition.data",
        'columns': (
            'lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr',
            'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx'),
        'classes_name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
        'question_mark_count': __default_question_mark_count,
        'x_slice': (slice(None, None), slice(1, None)),
        'y_slice': (slice(None, None), 0),
        'mlp_hidden_layers_sizes': [100, 10],
        'knn_k': __default_knn_k,
        'random_forest_estimator': __deafault_random_forest_estimator,
    },
    'poker': {
        'url': "./ts/poker-hand-testing.data",
        'columns': ('S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class'),
        'x_slice': (slice(None, None), slice(0, -1)),
        'y_slice': (slice(None, None), -1),
        'classes_name': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'question_mark_count': __default_question_mark_count,
        'mlp_hidden_layers_sizes': [20, 10],
        'knn_k': __default_knn_k,
        'random_forest_estimator': __deafault_random_forest_estimator,
    },
}

def main():

    # Script argument parsing
    parser = argparse.ArgumentParser(description='Homework 03 - Machine learning a.a. 2018/19 - Predict missing values',
                                                 epilog=' coded by: Emanuele Palombo')

    parser.add_argument('dataset_name', metavar='DATASET', type=str, nargs='?', default=__default_ts_name,
                        help='{} (default {}) - dataset name'.format(list(__ts_opts.keys()), __default_ts_name))

    parser.add_argument('--test-size', '-t', dest='test_size', action='store', metavar='TEST_SIZE', type=float, default=__default_test_size,
                        help='[0-1] (default {}) - splitting size of TestSet'.format(__default_test_size))

    parser.add_argument('--question-marks-ts', '-q', dest='qm_repeted_ts', action='store', type=int, default=__default_question_mark_count_repeated,
                        help='{{0,1,2...}} (default {}) - (this value * {} * samples) added to TrainingSet'.format(__default_question_mark_count_repeated,
                                                                                                               __default_question_mark_count))

    parser.add_argument('--no-split', '-s', dest='no_split', action='store_true', default=__default_no_split,
                        help='(default {}) - keep whole DataSet for training'.format(__default_no_split))

    parser.add_argument('--img-tag', '-i', dest='img_tag', action='store', type=str, default='',
                        help='string - add arbitrary string to saved images')

    parser.add_argument('--verbose', '-v', dest='verbosity', action='count', default=__default_training_verbosity,
                        help='add more verbosity to output (repeat it to increase)')

    args = parser.parse_args()

    if args.dataset_name not in __ts_opts:
        print('ERROR: Choose correct DataSet!\n')
        parser.print_help()
        exit(1)

    trainingset_selected_name = args.dataset_name
    test_size = args.test_size
    qm_repeted_ts = args.qm_repeted_ts
    dataset_no_split = args.no_split
    training_verbosity = args.verbosity
    img_tag = args.img_tag
    running_id = id_generator()

    ts_selected_opts = __ts_opts[trainingset_selected_name]
    # End script argument parsing

    print('\nDataSet selected: ' + ts_selected_opts['url'])

    # read dataset to pandas dataframe
    dataset = pd.read_csv(ts_selected_opts['url'], names=ts_selected_opts['columns'])

    if training_verbosity >= 1:
        print('\nFirst five rows of DataSet:\n')
        print(dataset.head())
        print('\nDataSet Length: {}'.format(len(dataset)))

    # DataSet Manipulation
    # remove row with question marks (this avoid to have '?' on the output)
    dataset = dataset[~(dataset.astype(str) == '?').any(1)]

    # strip out (remove) the "real output" (y)
    dataset = dataset.iloc[ts_selected_opts['x_slice'][0], ts_selected_opts['x_slice'][1]]

    # Different approach to value conversion
    # convert all column to int (str => int)
    # dataset = dataset.apply(lambda x: pd.factorize(x)[0] + 1)
    # convert all columns to int
    dataset = dataset.astype(int)

    # dataSet Information
    features_count = len(dataset.columns)
    features_values = ds_features_values(dataset)

    # copy input features to output (columns * 2)
    for column in dataset.columns:
        dataset['y_' + column] = dataset[column]

    # Split DataSet
    training_set, test_set = train_test_split(dataset, test_size=test_size,
                                                        random_state=__default_train_test_split_random_state)

    # check feature values between TrainingSet and TestSet
    # it's important avoid more value on TestSet (ie. error on log_loss for mismatch in predict_proba size)
    if not check_labels_split(features_count, training_set, test_set):
        exit(1)

    # Concat (add row) TrainingSet and TestSet
    # in this case model could see all sample (included queries without '?')
    if dataset_no_split:
        training_set = pd.concat([training_set, test_set], axis=0)

        print('\nTraining over the whole DataSet')
    else:
        print('\nSplit DataSet in TrainingSet and TestSet (test size: {})'.format(test_size))

    # add (append) question mark
    # append qm_count rows, with 1 to qm_count '?'
    qm_count = int(ts_selected_opts['question_mark_count'])
    for i in range(qm_repeted_ts):
        for value_count in range(1, qm_count + 1):
            training_set = ds_mod_with_value(training_set, value_count, features_count, True)

            if training_verbosity >= 1:
                print('{} Added {} question mark (?) to TrainingSet for each sample'.format(i, value_count))

    # Shuffle TrainingSet
    training_set = training_set.sample(frac=1)

    if training_verbosity >= 1:
        print('\nManipulated TrainingSet:\n')
        print(training_set.head())
        print('\nTrainingSet Length: {}'.format(len(training_set)))

    # TrainingSet: input X (features) and Output y ("mirrored" features))
    x_train = training_set.iloc[:, 0:features_count]
    y_train = training_set.iloc[:, features_count:]

    # TestSet: input X (features) and Output y ("mirrored" features))
    x_test = test_set.iloc[:, 0:features_count]
    y_test = test_set.iloc[:, features_count:]

    if training_verbosity >= 2:
        print('\nInput train:\n {}'.format(x_train.head()))
        print('\nOutput train:\n {}'.format(y_train.head()))
        print('\nInput test:\n {}'.format(x_test.head()))
        print('\nOutput test:\n {}'.format(y_test.head()))

    x_train = x_train.values
    y_train = y_train.values
    y_test = y_test.values

    # oneHot encoding (characteristic vector)
    # passing features_values without None force OneHotEncoder to transform None to null vector
    one_hot_encoder = OneHotEncoder(categories=features_values, handle_unknown='ignore')
    one_hot_encoder.fit(x_train)
    x_train_encoded = one_hot_encoder.transform(x_train).toarray()

    if training_verbosity >= 2:
        print('\nOneHotEncoding...\nexample: {} => {}'.format(x_train[0], x_train_encoded[0]))

    # store all results/metrics for each model/classifier
    results = {}

    for classifier_name in __deafult_model_classifier:

        filename = 'model_{}_{}.sav'.format(trainingset_selected_name, classifier_name)

        if os.path.isfile(filename):
            # load module already trained
            multi_output_classifier = joblib.load(filename)

            print('\n### Model {} loaded by file: {}\nImportant: remove the file to re-train the model!'
                  .format(classifier_name, filename))
        else:
            n_jobs=None
            model_verbosity = True if training_verbosity >= 3 else False
            
            if classifier_name == 'MLP':
                classifier = MLPClassifier(hidden_layer_sizes=ts_selected_opts['mlp_hidden_layers_sizes'],
                                           max_iter=1000, verbose=model_verbosity)
            elif classifier_name == 'KNN':
                n_jobs = None
                classifier = KNeighborsClassifier(n_neighbors=ts_selected_opts['knn_k'])
            elif classifier_name == 'SVM':
                classifier = SVC(gamma='scale', decision_function_shape='ovo', probability=True, verbose=model_verbosity)
            elif classifier_name == 'RandomForest':
                classifier = RandomForestClassifier(n_estimators=ts_selected_opts['random_forest_estimator'], verbose=model_verbosity)

            print('\n### Init and training the model: {}'.format(classifier_name))

            # init MultiOutput for classifier
            multi_output_classifier = MultiOutputClassifier(classifier, n_jobs=n_jobs)
            multi_output_classifier.fit(x_train_encoded, y_train)

            # save the model to disk
            joblib.dump(multi_output_classifier, filename)

        results[classifier_name] = collections.defaultdict(list)
        metris_result = results[classifier_name]

        # create input test (query) with different number of '?'
        for query_count_question_mark in range(ts_selected_opts['question_mark_count'] + 1):

            print('\n## Add {} questions mark to input test (query)'.format(query_count_question_mark))

            # modify (in place) input test with question marks
            x_test_with_qm = ds_mod_with_value(x_test.copy(), value_count=query_count_question_mark, append=False)

            if training_verbosity >= 2:
                print('\nInput test (query):\n {}'.format(pd.DataFrame(data=x_test_with_qm).head()))

            # encode the input test
            x_test_encoded = one_hot_encoder.transform(x_test_with_qm).toarray()

            # compute output prediction and probability
            y_pred = multi_output_classifier.predict(x_test_encoded)
            y_pred_proba = multi_output_classifier.predict_proba(x_test_encoded)
            # precision on whole output
            score = multi_output_classifier.score(x_test_encoded, y_test)
            # the Hamming loss corresponds to the Hamming distance between y_test and y_pred
            hamming_loss = np.sum(np.not_equal(y_test, y_pred))/float(y_test.size)

            # compute y_test and y_pred how if the out was only the query question marks
            y_test_reduced, y_pred_reduced = reduce_y_to_qm(x_test_with_qm, y_test, y_pred)

            # write y_pred_proba to file (csv)
            write_pred_proba(y_pred_proba, '{}{}-{}-q{}-{}{}.csv'.format(__default_csv_path,
                                                                        trainingset_selected_name,
                                                                        classifier_name,
                                                                        query_count_question_mark,
                                                                        running_id, img_tag))

            print('\nMetrics:')
            print(' {:<30} | {:^10} | {:>10}'.format('features', 'accuracy', 'log loss') )
            print('-'*(30+10+10+7))

            log_loss_avg = 0
            # for each output column => compute accuracy and log_loss
            for feature_index in range(y_test.shape[1]):
                y_test_column = y_test[:, feature_index]
                y_pred_column = y_pred[:, feature_index]

                accuracy = accuracy_score(y_test_column, y_pred_column)
                # note: for avoid error here was implemented check_labels_split()
                log_loss_value = log_loss(y_test_column, y_pred_proba[feature_index], labels=features_values[feature_index])

                print(' {:<30} | {:^10.4f} | {:>10.4f}'.format(test_set.columns[feature_index], accuracy, log_loss_value))

                log_loss_avg += log_loss_value

                metris_result['accuracy_' + str(feature_index)].append(accuracy)
                metris_result['log_loss_' + str(feature_index)].append(log_loss_value)

            print('\nVirtual reduced output:')
            # for each output reduced (only question marks) => compute accuracy
            for index in range(query_count_question_mark):
                accuracy = accuracy_score(y_test_reduced[:, index], y_pred_reduced[:, index])
                print(' accuracy {}:   {:>10.4f}'.format(index, accuracy))

                metris_result['accuracy_reduced_' + str(index)].append(accuracy)

            print('\nAll output:')
            print(' accuracy:     {:>10.4f}'.format(score))
            print(' log_loss avg: {:>10.4f}'.format(log_loss_avg/y_test.shape[1]))
            print(' hamming loss: {:>10.4f}'.format(hamming_loss))

            metris_result['accuracy'].append(score)
            metris_result['log_loss_avg'].append(log_loss_avg/y_test.shape[1])
            metris_result['hamming_loss'].append(hamming_loss)

        # GRAPH PLOT per model/classifier
        plot_line_graph(range(ts_selected_opts['question_mark_count'] + 1),
                        [results[classifier_name]['accuracy'], results[classifier_name]['log_loss_avg'], results[classifier_name]['hamming_loss']],
                        labels=['accuracy', 'log loss avg', 'hamming loss'],
                        fmt=['bo-', 'ro-', 'yo-'],
                        title=classifier_name,
                        xlabel='Number of Question Marks in the query',
                        ymax=1)

        if __default_save_img:
            plt.savefig('{}{}-{}-{}{}.png'.format(__default_imgs_path, trainingset_selected_name, classifier_name, running_id, img_tag), dpi=200)

        # create list of list of accuracy x feature
        accuracy_lst = ['accuracy_' + str(index) for index in range(features_count)]
        accuracy_lst = [results[classifier_name][accuracy_key] for accuracy_key in accuracy_lst]

        plot_line_graph(range(ts_selected_opts['question_mark_count'] + 1),
                        [results[classifier_name]['accuracy']] + accuracy_lst,
                        fmt=['bo-'] + ['g.--'] * len(accuracy_lst),
                        title=classifier_name + ': whole accuracy and those by features',
                        xlabel='Number of Question Marks in the query',
                        ylabel='accuracy',
                        ymax=1)

        if __default_save_img:
            plt.savefig('{}{}-{}-accuracy-{}{}.png'.format(__default_imgs_path, trainingset_selected_name, classifier_name, running_id, img_tag), dpi=200)

        # create list of list of accuracy_reduced x feature (adding 0 in front when needed)
        accuracy_reduced_lst = ['accuracy_reduced_' + str(index) for index in range(ts_selected_opts['question_mark_count'])]
        accuracy_reduced_lst = [results[classifier_name][accuracy_reduced] for accuracy_reduced in accuracy_reduced_lst]
        accuracy_reduced_lst = [ [None] * (ts_selected_opts['question_mark_count'] - len(accuracy_reduced) + 1) + accuracy_reduced
                                 for accuracy_reduced in accuracy_reduced_lst]

        plot_line_graph(range(ts_selected_opts['question_mark_count'] + 1),
                        [results[classifier_name]['accuracy']] + accuracy_reduced_lst,
                        fmt=['bo-'] + ['m.--'] * len(accuracy_reduced_lst),
                        title=classifier_name + ': whole accuracy and the virtual accuracies by features',
                        xlabel='Number of Question Marks in the query',
                        ylabel='accuracy',
                        ymax=1)

        if __default_save_img:
            plt.savefig('{}{}-{}-accuracy-reduced-{}{}.png'.format(__default_imgs_path, trainingset_selected_name, classifier_name, running_id, img_tag), dpi=200)

        # create list of list of log_loss x feature
        log_loss_lst = ['log_loss_' + str(index) for index in range(features_count)]
        log_loss_lst = [results[classifier_name][log_loss_key] for log_loss_key in log_loss_lst]

        plot_line_graph(range(ts_selected_opts['question_mark_count'] + 1),
                        [results[classifier_name]['log_loss_avg']] + log_loss_lst,
                        fmt=['ro-'] + ['c.--'] * len(log_loss_lst),
                        title=classifier_name + ': average log loss and those by features',
                        xlabel='Number of Question Marks in the query',
                        ylabel='log loss')

        if __default_save_img:
            plt.savefig('{}{}-{}-log-loss-{}{}.png'.format(__default_imgs_path, trainingset_selected_name, classifier_name, running_id, img_tag), dpi=200)

    metrics_by_classifier = [results[classifier][metric] for classifier in __deafult_model_classifier for metric in ['accuracy', 'log_loss_avg', 'hamming_loss']]
    label_by_classifier = [classifier + ' ' + metric for classifier in __deafult_model_classifier for metric in ['accuracy', 'log_loss_avg', 'hamming_loss']]
    fmt_lst = [style.replace('0', character) for character in ['o', '^', 'v', '<', '>', '.', ',', '+', 'x'] for style in ['b0-', 'r0-', 'y0-']]

    # GRAPH PLOT comparing model/classifier
    plot_line_graph(range(ts_selected_opts['question_mark_count'] + 1),
                    metrics_by_classifier,
                    labels=label_by_classifier,
                    fmt=fmt_lst,
                    title='Compare all model',
                    xlabel='Number of Question Marks in the query',
                    ylabel='',
                    ymax=1)

    if __default_save_img:
        plt.savefig('{}{}-comparing-{}{}.png'.format(__default_imgs_path, trainingset_selected_name, running_id, img_tag), dpi=200)

    if not __default_save_img:
        plt.show()

def ds_features_values(dataset):
    """ Get all features values (category x label) """
    features_values = []

    for value in dataset.columns:
        feature_values = dataset[value].unique().tolist()
        feature_values.sort()
        features_values.append(feature_values)

    return features_values

def ds_mod_with_value(dataset, value_count, index_limit=None, append=True, value=-9999):
    """ Edit/append value (eg. None) to random position of dataset row """
    if index_limit is None:
        index_limit = len(dataset.columns)

    for index, row in dataset.iterrows():
        row = row.copy()

        current_value_count = row.tolist().count(value)
        if (append and not current_value_count) or (not append and current_value_count < value_count):

            while row.tolist().count(value) < value_count:
                position = random.randrange(index_limit)

                row.loc[dataset.columns[position]] = value
            if append:
                dataset = dataset.append(row, ignore_index=True)
            else:
                dataset.loc[index] = row

    return dataset

def reduce_y_to_qm(x_test, y_test, y_pred, value=-9999):
    """ y_test and y_pred will be only on question marks"""
    y_test_reduced = []
    y_pred_reduced = []
    index = 0

    for x_test_index, row in x_test.iterrows():
        row = row.tolist()
        question_mark_index = [i for i, j in enumerate(row) if j == value]

        y_test_reduced.append(y_test[index, question_mark_index])
        y_pred_reduced.append(y_pred[index, question_mark_index])
        # note: it's not possible do the same with y_pred_proba because not all features have the
        # same number of class (tl;dr different row size for y_pred_proba_reduced)
        index += 1

    return np.array(y_test_reduced), np.array(y_pred_reduced)

def check_labels_split(features_count, training_set, test_set):
    """ check feature values between TrainingSet and TestSet
        it's important avoid more or different value on TestSet (ie. error on log_loss for mismatch in predict_proba size) """
    for feature_index in range(features_count):
        feature_values_training_set = np.sort(training_set.iloc[:, feature_index].unique())
        feature_values_test_set = np.sort(test_set.iloc[:, feature_index].unique())

        if feature_values_training_set.size < feature_values_test_set.size \
                or (feature_values_training_set.size == feature_values_test_set.size
                    and not np.array_equal(feature_values_training_set, feature_values_test_set)):

            print('\nERROR: There is a mismatch between Training and Test set values on feature: ' + training_set.columns[feature_index])
            print(' TrainingSet: \t{}'.format(np.sort(feature_values_training_set)))
            print(' TestSet: \t{}'.format(np.sort(feature_values_test_set)))

            print('This issue don\'t allow a correct test (ie. throw error). Please, change \'train_test_split_random_state\' until disapper')
            print('Test set must not have more values of Training!')

            return False

    return True

def write_pred_proba(pred_proba, filename):
    """ Write predicted probabilty to filename in CSV format """
    with open(filename, 'w') as f:
        n_features = len(pred_proba)

        for index_sample in range(len(pred_proba[0])):
            for index_feature in range(n_features):
                prob_vector_str = '"{}"'.format(pred_proba[index_feature][index_sample])
                prob_vector_str += ', ' if index_feature < (n_features - 1) else ''
                f.write(prob_vector_str)
            f.write('\n')

def plot_line_graph(x, ylist, fmt=('bo-', 'g^-', 'rv-', 'c>:', 'm<-'), labels=None, title='', xlabel='', ylabel='', ymax=None,
                    figsize=(12, 6)):
    """ Plot multiple line on single graph """
    plt.figure(figsize=figsize)
    if ymax is None:
        ymax = max([item if item else 0 for sublist in ylist for item in sublist])
    plt.ylim(top=ymax, bottom=min([item if item else 0 for sublist in ylist for item in sublist]))

    for index, y in enumerate(ylist):
        label = labels[index] if labels is not None else None
        plt.plot(x, y, fmt[index], label=label, linewidth=3, markersize=12)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    if labels:
        plt.legend(loc="best")

import string

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

import datetime

def get_time(seconds):
    return str(datetime.timedelta(seconds=seconds))

import time

if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s (running time) ---" % (get_time(time.time() - start_time)))

    exit(0)

