import os
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
stop_words = set(open(os.path.join(BASE_DIR, 'stopwords.txt'), 'r').read().split())


def clean_tweet(text):
    space_pattern = '\\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    rt_regex = '\\b[Rr][Tt]\\b'

    cleaned_tweet = re.sub(giant_url_regex, '_URL_', text)
    cleaned_tweet = re.sub(mention_regex, '_MTN_', cleaned_tweet)
    cleaned_tweet = re.sub(rt_regex, '', cleaned_tweet)
    cleaned_tweet = re.sub(space_pattern, ' ', cleaned_tweet)

    return cleaned_tweet


def clean_detox(text):
    space_pattern = '\\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    line_token_pattern = 'NEWLINE_TOKEN'

    cleaned_text = re.sub(giant_url_regex, '_URL_', text)
    cleaned_text = re.sub(line_token_pattern, ' ', cleaned_text)
    cleaned_text = re.sub(space_pattern, ' ', cleaned_text)

    return cleaned_text


def process_words(text):
    space_pattern = '\\s+'
    text = re.sub(space_pattern, ' ', text)

    words = text.split(' ')
    text = []
    for word in words:
        word = word.lower()
        if word not in stop_words:
            text.append(word)

    return ' '.join(text)
