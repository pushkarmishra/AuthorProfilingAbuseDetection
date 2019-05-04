import numpy
import os


class GraphFeatures:

    def __init__(self, CONFIG):
        self.BASE = CONFIG['BASE']
        self.EMBED_DIM = 200

        self.authors = {}
        with open(os.path.join(self.BASE, 'resources', 'authors.txt')) as authors:
            for line in authors.readlines():
                text_id, author_id = line.strip().split()
                self.authors[text_id] = author_id

        self.embeddings = {}
        with open(os.path.join(self.BASE, 'resources', 'authors.emb')) as embeds:
            for line in embeds.readlines():
                tokens = line.strip().split()
                author_id = tokens[0]
                embed = [float(x) for x in tokens[1:]]
                self.embeddings[author_id] = numpy.array(embed)


    def extract(self, text_id):
        author_id = self.authors.get(text_id, None)
        if author_id is None:
            return numpy.zeros(self.EMBED_DIM)

        return self.embeddings.get(author_id, numpy.zeros(self.EMBED_DIM))
