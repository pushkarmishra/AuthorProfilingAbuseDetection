from featureExtractor.feature_extractor import FeatureExtractor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

import coloredlogs
import logging
import numpy


logger = logging.getLogger('GridSearchLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


def gbc_details():
    classifier = LGBMClassifier(silent=False)
    parameters = {'num_leaves': [15, 31, 63, 127],
                  'min_child_weight': [1, 5, 7, 10, 20],
                  'min_child_samples': [1, 5, 10, 15, 20],
                  'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.25],
                  'n_estimators': [80, 100, 125, 150, 200]}
    return (classifier, parameters)


def lr_details():
    classifier = LogisticRegression(verbose=True, max_iter=1000)
    parameters = {'C': [0.01, 0.1, 0.25, 0.5, 0.75,
                        1.0, 10.0, 25.0, 50.0, 100.0]}
    return (classifier, parameters)


def perform_grid_search(texts_ids, all_texts, classes, args, CONFIG):
    estimator = args[0]
    size = CONFIG['GRID_SEARCH_SIZE']
    CONFIG['EMB_MODEL'] = args[1]
    CONFIG['NGRAM_MODEL'] = args[2]

    feature_extractor = FeatureExtractor(CONFIG)
    (classifier, parameters) = eval(estimator + '_details' + '()')

    data = []
    for (i, text) in enumerate(all_texts[:size]):
        features = feature_extractor.extract_features(text, texts_ids[i])
        data.append(features)

        if i % 1000 == 0 and i > 0:
            logger.info('{} of {} feature vectors prepared '
                        'for grid search'.format(i + 1, size))
    data = numpy.array(data)
    categories = numpy.array(classes[:size])

    clf = GridSearchCV(classifier, parameters, cv=5)
    clf.fit(data, categories)

    logger.info('Grid search results:\n{}'.format(clf.cv_results_))
    logger.info('Best param set: {}'.format(clf.best_params_))
