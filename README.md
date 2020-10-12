# Author Profiling for Abuse Detection

Code for paper "Author Profiling for Abuse Detection", in Proceedings of the 27th International Conference on Computational Linguistics (COLING) 2018

If you use this code, please cite our paper:
```
@inproceedings{mishra-etal-2018-author,
    title = "Author Profiling for Abuse Detection",
    author = "Mishra, Pushkar  and
      Del Tredici, Marco  and
      Yannakoudakis, Helen  and
      Shutova, Ekaterina",
    booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
    month = aug,
    year = "2018",
    address = "Santa Fe, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/C18-1093",
    pages = "1088--1098",
}
```

Python3.5+ required to run the code. Dependencies can be installed with `pip install -r requirements.txt` followed by `python -m nltk.downloader punkt`

The dataset for the code is provided in the _TwitterData/twitter_data_waseem_hovy.csv_ file as a list of _\[tweet ID, annotation\]_ pairs.
To run the code, please use a Twitter API (_twitter_access.py_ employs Tweepy) to retrieve the tweets for the given tweet IDs. Replace the dataset file with a
file of the same name that has a list of _\[tweet ID, tweet, annotation\]_ triples.
Additionally, _twitter_access.py_ contains functions to retrieve follower-following relationships amongst the authors of the tweets (specified in _resources/authors.txt_). Once the relationships have been retrieved, please use _Node2vec_ (see _resources/node2vec_) to produce embeddings for each of the authors and store them in a file named _authors.emb_ in the _resources_ directory.

To run the best method (LR + AUTH):
`python twitter_model.py -c 16202 -m lna`


<br/>To run the other methods:
* AUTH: `python twitter_model.py -c 16202 -m a`
* LR: `python twitter_model.py -c 16202 -m ln`
* WS: `python twitter_model.py -c 16202 -m ws`
* HS: `python twitter_model.py -c 16202 -m hs`
* WS + AUTH: `python twitter_model.py -c 16202 -m wsa`
* HS + AUTH: `python twitter_model.py -c 16202 -m hsa`

For the HS and WS based methods, adding the `-ft` flag to the command ensures that the pre-trained deep neural models from the _Models_ directory
are not used and instead all the training happens from scratch. This requires that the file of pre-trained GLoVe embeddings is downloaded from
<http://nlp.stanford.edu/data/glove.twitter.27B.zip>, unzipped and placed in the _resources_ directory prior to the execution.

<br/>An overview of the complete training-testing flow is as follows:
1. For each tweet in the dataset, its author's identity is obtained using functions available in the _twitter_access.py_ file. For each author,
information about which other authors from the dataset follow them on Twitter is also obtained in order to create a community graph where nodes
are authors and edges denote follow relationship.
2. Node2vec is applied to the community graph to generate embeddings for the nodes, i.e., the authors. These author embeddings are saved to the
_authors.emb_ file in the _resources_ directory.
3. The dataset is randomly split into train set and test set.
4. Tweets in the train set are used to produce an n-gram count based model or deep neural model depending on the method being used.
5. A feature extractor is instantiated that uses the models from step 2 along with the author embeddings to convert tweets to feature vectors.
6. LR/GBDT classifier is trained using the feature vectors extracted for the tweets in the train set. A part of the train set is held out as
validation data to prevent over-fitting.
7. The trained classifier is made to predict classes for tweets in the test set and precision, recall and F<sub>1</sub> are calculated.

In the 10-fold CV, steps 3-7 are run 10 times (each time with a different set of tweets as the test set) and the final precision, recall and
F<sub>1</sub> are calculated by averaging results from across the 10 runs.
