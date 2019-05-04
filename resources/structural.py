from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import PorterStemmer

import re


def word_stem(token):
    stem = PorterStemmer()
    return stem.stem(token)


def word_tokenizer(text):
    return word_tokenize(text)


def remove_non_words(all_words):
    only_words = []
    pattern = re.compile('[a-zA-Z]+')

    for word in all_words:
        if pattern.match(word) != None:
            only_words.append(word)
    return only_words


def sentence_tokenizer(text):
    return sent_tokenize(text)
