# from matplotlib import pyplot
from networkx.drawing.nx_agraph import write_dot
from tweepy import OAuthHandler

import coloredlogs
import csv
import json
import logging
import networkx
import os
import time
import tweepy


logger = logging.getLogger('TwitterAccess')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLEEP_TIME = 1000


class TwitterAccess:

    def __init__(self):
        self.api = self.load_api()


    def load_api(self):
        consumer_key = ''
        consumer_secret = ''
        access_token = ''
        access_secret = ''
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)

        # Load the twitter API via Tweepy
        return tweepy.API(auth)


    # Status Methods
    def tweet_text_from_tweet_id(self, idx):
        tweet = self.api.get_status(idx)
        return tweet.text


    # User Methods
    def get_followers(self, screen_name):
        user_ids = []
        for page in tweepy.Cursor(self.api.followers_ids,
                                  screen_name=screen_name).pages():
            user_ids.extend(page)
            time.sleep(60)

        return user_ids


    def user_from_tweet_id(self, idx):
        status = self.api.get_status(idx)
        return (status.user.id_str, status.user.screen_name)


    def get_follow_info(self, x, y):
        return self.api.show_friendship(source_id=x, target_id=y)


    def username_from_user_id(self, idx):
        user = self.api.get_user(user_id=idx)
        return user.screen_name


    def timeline_from_username(self, screen_name):
        timeline = self.api.user_timeline(screen_name=screen_name)
        return timeline


class Graph:

    def __init__(self, tweet_ids=None, nodes_file=None, edges_file=None):
        self.TWEET_IDS = tweet_ids
        self.ACCESSOR = TwitterAccess()
        self.GRAPH = networkx.Graph()
        self.NODES = {}
        self.EDGES = set()

        if nodes_file is not None:
            with open(os.path.join(BASE_DIR, 'resources', nodes_file)) as nodes:
                self.NODES = json.load(nodes)
        else:
            self.prepare_nodes()

        if edges_file is not None:
            with open(os.path.join(BASE_DIR, 'resources', edges_file)) as edges:
                for line in edges.readlines():
                    x, y = line.strip().split(',')
                    self.EDGES.add((x, y))


    def prepare_nodes(self):
        if self.TWEET_IDS is None:
            return

        def fill_node_data(idx):
            user = self.ACCESSOR.user_from_tweet_id(idx)
            self.NODES[user[0]] = user[1]

        for idx in self.TWEET_IDS:
            try:
                fill_node_data(idx)
            except tweepy.error.RateLimitError:
                try:
                    logger.info('Hit rate limit; waiting and retrying')
                    time.sleep(SLEEP_TIME)
                    fill_node_data(idx)
                except:
                    break
            except Exception as e:
                logger.error('Problem with tweet id {}: {}'.format(idx, e))

        with open(os.path.join(BASE_DIR, 'resources',
                               'authors.json'), 'w') as nodes_file:
            json.dump(self.NODES, nodes_file)


    def add_follower_edges(self):
        edges_file = open(os.path.join(BASE_DIR, 'resources',
                                       'author_edges.txt'), 'a')
        def fill_edge_data(x):
            followers =  self.ACCESSOR.get_followers(self.NODES[x])
            followers = set([str(f) for f in followers])

            for y in self.NODES:
                if (x, y) in self.EDGES:
                    continue

                if y in followers:
                    self.EDGES.add((y, x))
                    print('{},{}'.format(y, x), file=edges_file)
            edges_file.flush()
            logger.info('Followers of user {} added'.format(self.NODES[x]))

        for x in self.NODES.keys():
            try:
                fill_edge_data(x)
            except tweepy.error.RateLimitError:
                try:
                    logger.info('Hit rate limit; waiting and retrying')
                    time.sleep(SLEEP_TIME)
                    fill_edge_data(x)
                except:
                    break
            except Exception as e:
                logger.error('Problem with user {}: {}'.format(self.NODES[x], e))
        edges_file.close()


    def form_graph(self):
        for node_id in self.NODES.keys():
            self.GRAPH.add_node(self.NODES[node_id])
        for edge in self.EDGES:
            self.GRAPH.add_edge(self.NODES[edge[0]], self.NODES[edge[1]])


    def print_graph(self):
        write_dot(self.GRAPH, 'graph.dot')
        # networkx.draw(self.GRAPH)
        # pyplot.savefig('graph.png')


def main():
    f = open(os.path.join(BASE_DIR, 'TwitterData', 'twitter_data_waseem_hovy.csv'),
             'r', encoding='utf-8')
    csv_read = csv.reader(f)

    count = 0
    tweet_ids = []
    for line in csv_read:
        count += 1
        if count == 1:
            continue

        idx, text, cat = line
        tweet_ids.append(idx)

    graph = Graph(tweet_ids=None, nodes_file='authors.json',
                  edges_file='author_edges.txt')
    graph.add_follower_edges()
    graph.form_graph()
    # graph.print_graph()


if __name__ == "__main__":
    main()
