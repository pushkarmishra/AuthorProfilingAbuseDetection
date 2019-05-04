from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

import datetime
import logging
import os


class NGramFeatures:

    def __init__(self, CONFIG):
        self.USE_IDF = CONFIG['TF_USE_IDF']
        self.NRANGE = CONFIG['TF_NRANGE']
        self.SUBLIN = CONFIG['TF_SUBLIN']
        self.MAX_FEAT = CONFIG['TF_MAX_FEAT']
        self.BASE = CONFIG['BASE']

        self.model = None
        if CONFIG['NGRAM_MODEL'] is not None:
            self.model = joblib.load(os.path.join(self.BASE, 'Models',
                                                  CONFIG['NGRAM_MODEL']))


    def extract(self, text):
        return self.model.transform([text]).toarray()[0]


    def train(self, all_texts):
        self.model = TfidfVectorizer(analyzer='char',
                                     ngram_range=self.NRANGE,
                                     max_features=self.MAX_FEAT,
                                     use_idf=self.USE_IDF,
                                     sublinear_tf=self.SUBLIN)
        self.model.fit(all_texts)

        # Save N-gram vocabulary
        cur_time = str(datetime.datetime.now()).replace(':', '-') \
                                                .replace(' ', '_')
        model_name = 'NGramModel_' + cur_time + '.pkl'
        joblib.dump(self.model, os.path.join(self.BASE, 'Models', model_name))

        logger = logging.getLogger('TrainingLog')
        logger.info('N-gram vectorization finished with vocabulary'
                    ' size {}'.format(len(self.model.vocabulary_)))

        return model_name
