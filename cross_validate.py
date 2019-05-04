from main_classifier import MainClassifier
from sklearn.model_selection import StratifiedKFold
from test import test

import coloredlogs
import logging


logger = logging.getLogger('CVLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')

EMB_MODEL = [
    'Emb_2018-03-04_12-22-03.453692.h5',
    'Emb_2018-03-04_12-29-57.342629.h5',
    'Emb_2018-03-04_12-38-56.418197.h5',
    'Emb_2018-03-04_12-46-41.840651.h5',
    'Emb_2018-03-04_12-54-29.838667.h5',
    'Emb_2018-03-04_13-02-14.060916.h5',
    'Emb_2018-03-04_13-09-58.910309.h5',
    'Emb_2018-03-04_13-17-44.565754.h5',
    'Emb_2018-03-04_13-25-30.865847.h5',
    'Emb_2018-03-04_13-33-38.104125.h5',
]

def run_cv(text_ids, all_texts, categories, CONFIG, folds=10):
    logger.info('{}-fold cross validation procedure has begun'.format(folds))

    k_fold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
    metrics = []
    count = 0
    for train_idx, test_idx in k_fold.split(all_texts, categories):
        count += 1
        logger.info('Validation round {} of {} starting'
                    .format(count, folds))

        ids_train, X_train, y_train = [], [], []
        for idx in train_idx:
            ids_train.append(text_ids[idx])
            X_train.append(all_texts[idx])
            y_train.append(categories[idx])

        ids_test, X_test, y_test = [], [], []
        for idx in test_idx:
            ids_test.append(text_ids[idx])
            X_test.append(all_texts[idx])
            y_test.append(categories[idx])

        if CONFIG['EMB_MODEL'] is None:
            CONFIG['EMB_MODEL'] = EMB_MODEL[count - 1]
        else:
            CONFIG['EMB_MODEL'] = None

        classifier = MainClassifier(CONFIG)
        classifier.train(ids_train, X_train, y_train)

        metrics.append(test(ids_test, X_test, y_test, classifier))

    # Average metrics
    logger.info('\n')
    logger.info('Summary (precision, recall, F1, accuracy):')

    prec = rec = f1 = acc = 0.0
    for (i, metric) in enumerate(metrics):
        logger.info('Metrics for round {}: {}'.format(i + 1, metric))
        prec += metric[0]
        rec += metric[1]
        f1 += metric[2]
        acc += metric[3]

    logger.info('\n')
    logger.info('Final average metrics: {}, {}, {}, {}'.format(prec/folds,
                                                               rec/folds,
                                                               f1/folds,
                                                               acc/folds))
