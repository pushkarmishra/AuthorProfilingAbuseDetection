from main_classifier import MainClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

import coloredlogs
import logging
import numpy


logger = logging.getLogger('TestLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


def one_hot(y):
    m = y.shape[0]

    if len(y.shape) == 1:
        n = len(set(y.ravel()))
        idxs = y.astype(int)
    else:
        idxs = y.argmax(axis=1)
        n = y.shape[1]

    y_oh = numpy.zeros((m, n))
    y_oh[list(range(m)), idxs] = 1

    return y_oh


def compute_roc_auc(classes, probs):
    classes_arr = one_hot(numpy.array(classes))
    prob_arr = numpy.array(probs)

    return roc_auc_score(classes_arr, prob_arr, average='macro')


def test(text_ids, texts, classes, classifier):
    classes_pred = []
    probs = []
    count_match = 0
    for (i, text) in enumerate(texts):
        (clazz, prob_score) = classifier.classify(text_ids[i], text, prob=True)
        probs.append(prob_score)
        classes_pred.append(clazz)
        if clazz == classes[i]:
            count_match += 1

        if i > 0 and i % 100 == 0:
            accuracy = (1.0 * count_match) / (i + 1)
            logger.info('{} samples classified. Accuracy up till '
                        'now is {}'.format(i + 1, accuracy))

    # Calculate metrics
    accuracy = (1.0 * count_match) / len(classes)
    report = classification_report(classes, classes_pred, digits=5)
    conf_matrix = confusion_matrix(classes, classes_pred)
    roc_auc = compute_roc_auc(classes, probs)

    # Log results
    logger.info('Total {} samples classified with accuracy '
                '{}'.format(len(classes), accuracy))
    logger.info('AUROC is {}'.format(roc_auc))
    logger.info('Classification report:\n{}'.format(report))
    logger.info('Confusion matrix:\n{}'.format(conf_matrix))

    metrics = precision_recall_fscore_support(classes, classes_pred,
                                              average='weighted')
    metrics = [metrics[0], metrics[1], metrics[2],
               accuracy_score(classes, classes_pred)]

    return metrics
