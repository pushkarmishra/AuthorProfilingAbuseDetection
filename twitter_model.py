import numpy
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(57)
random.seed(75)
os.environ['KERAS_BACKEND'] = 'theano'

if os.environ['KERAS_BACKEND'] == 'tensorflow':
    import tensorflow
    tensorflow.set_random_seed(35)

from cross_validate import run_cv
from grid_search import perform_grid_search
from main_classifier import MainClassifier
from resources.textual import clean_tweet
from test import test

import argparse
import csv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'EMB_FILE': 'glove.twitter.27B.200d.txt',
    'EMB_MODEL': None,
    'EMB_DIM': 200,
    'EMB_MIN_DF': 1,
    'EMB_MAX_DF': -1,
    'EMB_MAX_VCB': 50000,
    'WORD_MIN_FREQ': 2,
    'DNN_EPOCH': 50,
    'DNN_BATCH': 64,
    'DNN_VAL_SPLIT': 0.04,
    'DNN_HIDDEN_UNITS': 128,
    'GB_LEAVES': 31,
    'GB_LEAF_WEIGHT': 7,
    'GB_LEAF_SAMPLES': 10,
    'GB_ITERATIONS': 125,
    'GB_LEARN_RATE': 0.08,
    'LR_C': 25,
    'NGRAM_MODEL': None,
    'TF_NRANGE': (1, 4),
    'TF_SUBLIN': False,
    'TF_MAX_FEAT': 10000,
    'TF_USE_IDF': False,
    'CLASSIFIER': None,
    'METHOD': None,
    'GRID_SEARCH_SIZE': 25000,
    'BASE': BASE_DIR,
}


def read_data(data_file):
    read_f = open(data_file, 'r', encoding='utf-8')
    csv_read = csv.reader(read_f)

    texts = []
    classes = []
    ids = []
    count = 0

    for line in csv_read:
        count += 1
        if count == 1:
            continue

        id, text, clazz = line
        classes.append(int(clazz))
        texts.append(text)
        ids.append(id)

    return (ids, texts, classes)


def check_classifier():
    classifier = MainClassifier(CONFIG)
    classifier.classify(None, '')
    while(True):
        text = input()
        category = classifier.classify(None, text)
        print(category)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Experimentation with'
                                                 ' Twitter datasets')

    parser.add_argument('-c', '--cross_val', action='store', type=int,
                        dest='cross_val_size',
                        help='Part of dataset to be used for cross validation')

    parser.add_argument('-g', '--grid_search', action='store', type=str,
                        nargs=3, dest='grid_params',
                        metavar=('ESTIMATOR: gbc/svm', 'FEATURES', 'FEATURES'),
                        help='Model and features to be used for grid search')

    parser.add_argument('-t', '--train_test', action='store', type=int,
                        dest='train_test_split', default=10000,
                        help='Split point of data for training and testing')

    parser.add_argument('-m', '--method', action='store', type=str,
                        dest='method', default='lna',
                        help='Method to run')

    parser.add_argument('-ft', '--full-train', action='store_true',
                        dest='full_train',
                        help='Presence of flag will ensure pre-trained '
                             'models are not used')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    data_file = os.path.join(BASE_DIR, 'TwitterData', 'twitter_data_waseem_hovy.csv')
    (ids, texts, classes) = read_data(data_file)
    texts = [clean_tweet(t) for t in texts]

    CONFIG['METHOD'] = args.method
    CONFIG['EMB_MODEL'] = '' if args.full_train else None
    if args.cross_val_size is not None:
        run_cv(ids[:args.cross_val_size],
               texts[:args.cross_val_size],
               classes[:args.cross_val_size],
               CONFIG)

    elif args.grid_params is not None:
        perform_grid_search(ids, texts, classes, args.grid_params, CONFIG)

    else:
        classifier = MainClassifier(CONFIG)

        split = args.train_test_split
        classifier.train(ids[:split], texts[:split], classes[:split])
        test(ids[split:], texts[split:], classes[split:], classifier)
