from featureExtractor.dnn_features import DNNFeatures
from featureExtractor.feature_extractor import FeatureExtractor
from featureExtractor.ngram_features import NGramFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import coloredlogs
import copy
import datetime
import lightgbm
import logging
import numpy
import os


logger = logging.getLogger('TrainingLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class MainClassifier:

    def __init__(self, CONFIG):
        self.CONFIG = copy.deepcopy(CONFIG)
        self.BASE = CONFIG['BASE']

        self. featureExtract = None
        self.classifier = None
        if CONFIG['CLASSIFIER'] is not None:
            self.classifier = joblib.load(os.path.join(self.BASE, 'Models',
                                                       CONFIG['CLASSIFIER']))


    def train(self, text_ids, all_texts, classes):
        logger = logging.getLogger('TrainingLog')
        logger.info('Initiating training of main classifier')

        # Prepare feature extractor
        if self.CONFIG['EMB_MODEL'] is None and \
                ('ws' in self.CONFIG['METHOD'] or 'hs' in self.CONFIG['METHOD']):
            self.CONFIG['EMB_MODEL'] = \
                DNNFeatures(self.CONFIG).train(all_texts, classes)

        if self.CONFIG['NGRAM_MODEL'] is None and 'n' in self.CONFIG['METHOD']:
            self.CONFIG['NGRAM_MODEL'] = \
                NGramFeatures(self.CONFIG).train(all_texts)

        self.featureExtract = FeatureExtractor(self.CONFIG)
        logger.info('Feature extractor ready')

        # Prepare data
        data = []
        for (i, text) in enumerate(all_texts):
            features = self.featureExtract.extract_features(text, text_ids[i])
            data.append(features)

            if i % 1000 == 0 and i > 0:
                logger.info('{} of {} feature vectors prepared '
                            'for training'.format(i + 1, len(all_texts)))
        train_X, train_Y = numpy.array(data), numpy.array(classes)

        # Train classifier
        train_data = lightgbm.Dataset(train_X, train_Y)
        params = {
            'learning_rate': self.CONFIG['GB_LEARN_RATE'],
            'num_leaves': self.CONFIG['GB_LEAVES'],
            'min_child_weight': self.CONFIG['GB_LEAF_WEIGHT'],
            'min_child_samples': self.CONFIG['GB_LEAF_SAMPLES'],
            'objective': 'multiclass',
            'num_class': len(set(classes)),
            'metric': {'multi_logloss'},
        }
        if 'l' not in self.CONFIG['METHOD']:
            self.classifier = lightgbm.train(params, train_data,
                                             self.CONFIG['GB_ITERATIONS'])
        else:
            self.classifier = LogisticRegression(C=self.CONFIG['LR_C'])
            self.classifier.fit(train_X, train_Y)

        # Save classifier
        cur_time = str(datetime.datetime.now()).replace(':', '-') \
                                                .replace(' ', '_')
        self.CONFIG['CLASSIFIER'] = 'Classifier_' + cur_time + '.pkl'
        joblib.dump(self.classifier, os.path.join(self.BASE, 'Models',
                                                  self.CONFIG['CLASSIFIER']))

        logger = logging.getLogger('TrainingLog')
        logger.info('Main classifier training finished')

        return self.CONFIG['CLASSIFIER']


    def classify(self, text_id, text, prob=False):
        # Prepare classifier
        if self.classifier is None:
            logger = logging.getLogger('TrainingLog')
            models = os.listdir(os.path.join(self.BASE, 'Models'))
            models.sort(reverse=True)

            for model in models:
                if model.startswith('Classifier') and model.endswith('.pkl'):
                    self.CONFIG['CLASSIFIER'] = model
                    break

            logger.info('Using Classifier Model {}'
                        .format(self.CONFIG['CLASSIFIER']))
            self.classifier = joblib.load(os.path.join(self.BASE, 'Models',
                                          self.CONFIG['CLASSIFIER']))

        # Prepare feature extractor
        if self.featureExtract is None:
            logger = logging.getLogger('TrainingLog')
            models = os.listdir(os.path.join(self.BASE, 'Models'))
            models.sort(reverse=True)

            if self.CONFIG['EMB_MODEL'] is None:
                for model in models:
                    if model.startswith('Emb_') and model.endswith('.h5'):
                        self.CONFIG['EMB_MODEL'] = model
                        break

            if self.CONFIG['NGRAM_MODEL'] is None:
                for model in models:
                    if model.startswith('NGram') and model.endswith('.pkl'):
                        self.CONFIG['NGRAM_MODEL'] = model
                        break

            logger.info('Using Embedding Model {} and N-gram Model {}'
                        .format(self.CONFIG['EMB_MODEL'],
                                self.CONFIG['NGRAM_MODEL']))

            self.featureExtract = FeatureExtractor(self.CONFIG)
            logger.info('Feature extractor ready')

        # Classify
        features = self.featureExtract.extract_features(text, text_id)
        features = numpy.array([features])
        if isinstance(self.classifier, LogisticRegression):
            prediction = self.classifier.predict_proba(features)[0].tolist()
        else: prediction = self.classifier.predict(features)[0].tolist()

        if prob:
            return (prediction.index(max(prediction)), prediction)
        return prediction.index(max(prediction))
