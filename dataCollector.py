import json
import os
import re
import string
import time
from os import listdir

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

import config


def fetch_init_counter(data_dir):
    heighest_index = 0
    for filename in listdir(data_dir):
        # Get index of filename
        file_index = int(filename.split('_')[1].split('.')[0])
        # Check if the file_index is heigher than the current heighest index
        if file_index > heighest_index:
            heighest_index = file_index

    return heighest_index


class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""

    def __init__(self, data_dir, sentiment):
        self.data_dir = data_dir
        self.counter = fetch_init_counter(data_dir)
        self.sentiment = sentiment
        filename = "%s/" + sentiment + "_%s.txt"

        self.current_filename = filename % (data_dir, str(self.counter))

    def on_data(self, data):
        try:
            self.counter += 1
            filename = "%s/" + self.sentiment + "_%s.txt"
            self.current_filename = filename % (self.data_dir, str(self.counter))
            with open(self.current_filename, 'a', encoding="utf-8") as f:
                formatted_data = format_tweet(data)
                f.write(formatted_data)
                print(self.current_filename, ":", formatted_data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))

            os.remove(self.current_filename)
            self.counter -= 1

            time.sleep(5)
        return True

    def on_error(self, status):
        print(status)
        return True


def de_emojify(emojified_tweet):
    cleared_tweet = ""
    for character in emojified_tweet:
        try:
            character.encode("ascii")
            cleared_tweet += character
        except UnicodeEncodeError:
            cleared_tweet += ''
    return cleared_tweet


def clean_tweet(tweet):
    # remove http links
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove Emojis etc.
    tweet = de_emojify(tweet)
    # Remove mentions and retweets
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    # Remove Retweet RT
    tweet = re.sub(r'RT+\s', '', tweet)
    # remove cutted words
    tweet = re.sub(r'\w+(...)$', '', tweet)
    # remove punctuation from tweet
    translator = str.maketrans('', '', string.punctuation)
    tweet = tweet.translate(translator)
    # remove leading whitespaces
    tweet = tweet.lstrip(' ')
    # Format the tweet to lowercase
    tweet = tweet.lower()

    return tweet


def format_tweet(data):
    json_object = json.loads(data)

    tweet = json_object['text']

    return clean_tweet(tweet)


def format_filename(fname):
    """Convert file name into a safe string.
    Arguments:
        fname -- the file name to convert
    Return:
        String -- converted file name
    """
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    """Convert a character into '_' if invalid.
    Arguments:
        one_char -- the char to convert
    Return:
        Character -- converted char
    """
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'


def stream_pos_tweets(auth):
    data_dir = "twitterdata/pos"
    pos_query = [":)", ":-)"]

    twitter_stream = Stream(auth, MyListener(data_dir, "pos"))
    twitter_stream.filter(track=pos_query, languages=['en'])


def stream_neg_tweets(auth):
    data_dir = "twitterdata/neg"
    neg_query = [":(", ":-("]

    twitter_stream = Stream(auth, MyListener(data_dir, sentiment="neg"))
    twitter_stream.filter(track=neg_query, languages=['en'])


def setup_authenticator():
    auth = OAuthHandler(config.consumer_key, config.consumer_secret)
    auth.set_access_token(config.access_token, config.access_token_secret)
    return auth


auth = setup_authenticator()

# stream_pos_tweets(auth)

stream_neg_tweets(auth)
