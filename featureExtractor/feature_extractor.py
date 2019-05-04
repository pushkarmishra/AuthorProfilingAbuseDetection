from featureExtractor.dnn_features import DNNFeatures
from featureExtractor.graph_features import GraphFeatures
from featureExtractor.ngram_features import NGramFeatures


class FeatureExtractor:

    def __init__(self, CONFIG):
        self.METHOD = CONFIG['METHOD']

        if 'hs' in self.METHOD or 'ws' in self.METHOD:
            self.dnn = DNNFeatures(CONFIG)
        if 'n' in self.METHOD:
            self.ngram = NGramFeatures(CONFIG)
        if 'a' in self.METHOD:
            self.graph = GraphFeatures(CONFIG)


    def extract_features(self, text, text_id=None):
        features = []

        if 'hs' in self.METHOD or 'ws' in self.METHOD:
            self.get_dnn_features(features, text)
        if 'n' in self.METHOD:
            self.get_ngram_features(features, text)
        if 'a' in self.METHOD:
            self.get_graph_features(features, text_id)

        return features


    def get_dnn_features(self, features, text):
        if 'ws' in self.METHOD:
            features += self.dnn.sum_word_embeddings(text).tolist()
        else:
            features += self.dnn.last_hidden_state(text).tolist()


    def get_ngram_features(self, features, text):
        features += self.ngram.extract(text).tolist()


    def get_graph_features(self, features, text_id):
        features += self.graph.extract(text_id).tolist()
