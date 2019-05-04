from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import Input
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from resources.textual import process_words
from resources.structural import word_tokenizer
from string import punctuation

import datetime
import io
import logging
import numpy
import os
import pickle
import re
import time


class DNNFeatures:

    def __init__(self, CONFIG):
        self.EMBED_DIM = CONFIG['EMB_DIM']
        self.EMB_FILE = CONFIG['EMB_FILE']
        self.MIN_DF = CONFIG['EMB_MIN_DF']
        self.MAX_DF = CONFIG['EMB_MAX_DF']
        self.MAX_VOCAB = CONFIG['EMB_MAX_VCB']
        self.WORD_MIN_FREQ = CONFIG['WORD_MIN_FREQ']
        self.EPOCH = CONFIG['DNN_EPOCH']
        self.BATCH_SIZE = CONFIG['DNN_BATCH']
        self.VAL_SPLIT = CONFIG['DNN_VAL_SPLIT']
        self.HIDDEN_UNITS = CONFIG['DNN_HIDDEN_UNITS']
        self.BASE = CONFIG['BASE']

        self.model = None
        self.prediction_model = None
        self.vocab = None
        self.word_freq = None

        if CONFIG['EMB_MODEL'] is not None:
            saved_vocab = CONFIG['EMB_MODEL'].split('.h5')[0] + '.pkl'
            self.model = load_model(os.path.join(self.BASE, 'Models',
                                                 CONFIG['EMB_MODEL']))

            with open(os.path.join(self.BASE, 'Models', saved_vocab), 'rb') as vocab_file:
                self.vocab, self.word_freq = pickle.load(vocab_file)


    def tokenize_text(self, texts):
        text_tokens = []
        for (i, text) in enumerate(texts):
            text = re.sub('[' + punctuation + ']', ' ', text)
            text = re.sub('\\b[0-9]+\\b', '', text)
            text = process_words(text)

            tokens = word_tokenizer(text)
            text_tokens.append(tokens)

        return text_tokens


    def build_vocab(self, text_tokens):
        self.word_freq = {}
        for text_token in text_tokens:
            for token in set(text_token):
                self.word_freq[token] = self.word_freq.get(token, 0) + 1
        self.word_freq = [(f, w) for (w, f) in self.word_freq.items()]
        self.word_freq.sort(reverse=True)

        token_counts = []
        for (count, token) in self.word_freq:
            if self.MAX_DF != -1 and count > self.MAX_DF:
                continue
            if count < self.MIN_DF:
                continue
            token_counts.append((count, token))

        token_counts.sort(reverse=True)
        if self.MAX_VOCAB != -1:
            token_counts = token_counts[:self.MAX_VOCAB]
        # NIV: not in vocab token, i.e., out of vocab
        token_counts.append((0, 'NIV'))

        self.vocab = {}
        for (i, (count, token)) in enumerate(token_counts):
            self.vocab[token] = i + 1


    def transform_texts(self, text_tokens):
        transformed = []
        for text_token in text_tokens:
            entry = []
            for token in text_token:
                entry.append(self.vocab.get(token, self.vocab['NIV']))
            transformed.append(entry)

        return transformed


    def prepare_model(self, emb_dimension, seq_length, num_categories):
        input = Input(shape=(seq_length,), dtype='int32')
        embed = Embedding(input_dim=len(self.vocab) + 1,
                          output_dim=emb_dimension,
                          input_length=seq_length,
                          mask_zero=True, trainable=True)(input)
        dropout_1 = Dropout(0.25)(embed)
        gru_1 = GRU(self.HIDDEN_UNITS, return_sequences=True)(dropout_1)
        dropout_2 = Dropout(0.25)(gru_1)
        gru_2 = GRU(self.HIDDEN_UNITS)(dropout_2)
        dropout_3 = Dropout(0.50)(gru_2)
        softmax = Dense(num_categories, activation='softmax')(dropout_3)

        self.model = Model(inputs=input, outputs=softmax)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])


    def train(self, texts, classes):
        logger = logging.getLogger('TrainingLog')

        tokens = self.tokenize_text(texts)
        self.build_vocab(tokens)
        logger.info('Vocabulary of size {} built for embeddings'
                    .format(len(self.vocab)))

        X = self.transform_texts(tokens)
        X = pad_sequences(X)

        seq_length = X.shape[1]
        class_weights = {}
        for clazz in classes:
            class_weights[clazz] = class_weights.get(clazz, 0) + 1
        for clazz in class_weights:
            class_weights[clazz] /= (1.0 * len(classes))

        y = numpy.array(classes)
        y = np_utils.to_categorical(y, len(class_weights))

        self.prepare_model(self.EMBED_DIM, seq_length, len(class_weights))
        if self.EMB_FILE is not None:
            trained_vectors = self.initialise_embeddings(
                                os.path.join(self.BASE, 'resources', self.EMB_FILE))
            self.model.layers[1].set_weights([trained_vectors])

        # Train DNN
        best_model = 'Emb_best_' + str(time.time()) + '.h5'
        checkpoint = ModelCheckpoint(os.path.join(self.BASE, 'Models', best_model),
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='auto')
        earlyStopping = EarlyStopping(monitor='val_loss',
                                      patience=3, verbose=0,
                                      mode='auto')
        callbacks = [checkpoint, earlyStopping]

        self.model.fit(X, y, epochs=self.EPOCH,
                       class_weight=class_weights,
                       batch_size=self.BATCH_SIZE,
                       validation_split=self.VAL_SPLIT,
                       callbacks=callbacks, verbose=2)
        self.model.load_weights(os.path.join(self.BASE, 'Models', best_model))

        # Save model
        logger.info('DNN training finished')
        cur_time = str(datetime.datetime.now()).replace(':', '-') \
                                                .replace(' ', '_')
        model_name = 'Emb_' + cur_time + '.h5'
        self.model.save(os.path.join(self.BASE, 'Models', model_name))
        vocab_name = 'Emb_' + cur_time + '.pkl'
        with open(os.path.join(self.BASE, 'Models', vocab_name), 'wb') as vocab_file:
            pickle.dump([self.vocab, self.word_freq], vocab_file)

        return model_name


    def sum_word_embeddings(self, text):
        tokens = self.tokenize_text([text])
        X = self.transform_texts(tokens)[0]

        embed = numpy.zeros(self.EMBED_DIM)
        embeddings = self.model.layers[1].get_weights()[0]

        for (i, word) in enumerate(X):
            embed += embeddings[word]
        embed = np_utils.normalize(embed)[0]

        return embed


    def last_hidden_state(self, text):
        if self.prediction_model is None:
            self.prediction_model = Model(inputs=self.model.input,
                                          outputs=self.model.layers[-3].output)

        tokens = self.tokenize_text([text])
        indexes = self.transform_texts(tokens)[0]
        seq_length = self.model.layers[1].input_length

        while len(indexes) < seq_length:
            indexes.append(0)
        indexes = indexes[:seq_length]

        X = numpy.array([indexes])
        return self.prediction_model.predict(X)[0]


    def predict(self, text):
        if self.prediction_model is None:
            self.prediction_model = Model(inputs=self.model.input,
                                          outputs=self.model.output)

        tokens = self.tokenize_text([text])
        indexes = self.transform_texts(tokens)[0]
        seq_length = self.model.layers[1].input_length

        while len(indexes) < seq_length:
            indexes.append(0)
        indexes = indexes[:seq_length]

        X = numpy.array([indexes])
        return self.prediction_model.predict(X)


    def initialise_embeddings(self, filename):
        logger = logging.getLogger('TrainingLog')
        weights = numpy.random.uniform(size=(len(self.vocab) + 1,
                                             self.EMBED_DIM),
                                       low=-0.05, high=0.05)

        with io.open(filename, 'r', encoding='utf-8') as vectors:
            for vector in vectors:
                tokens = vector.split(' ')
                word = tokens[0]
                embed = [float(val) for val in tokens[1:]]

                if word not in self.vocab:
                    continue
                weights[self.vocab[word]] = numpy.array(embed)
        logger.info('{} vectors initialised'.format(len(self.vocab)))

        return weights
