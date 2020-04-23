import nltk
import json


def get_tweets_for_model(tokens_list):
    # converted to dictionary data type
    for tweet_tokens in tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def word_tokenizer(data):
    token = []
    for item in data:
        token.append(nltk.word_tokenize(item))
    return token


# get data from json files
with open('negative_tweets.json') as n:
    negative_tweets = json.load(n)
    # get node tweets
    negative_tweets = negative_tweets["tweets"]
with open('positive_tweets.json') as p:
    positive_tweets = json.load(p)
    positive_tweets = positive_tweets["tweets"]

# tokenize the words in files
positive_tweet_tokens = word_tokenizer(positive_tweets)
negative_tweet_tokens = word_tokenizer(negative_tweets)

# converted the words in datatype of dictionary as required for the classifier
positive_tokens_for_model = get_tweets_for_model(positive_tweet_tokens)
negative_tokens_for_model = get_tweets_for_model(positive_tweet_tokens)
# assigned positive to all positive sentiments and negative to all negative sentiments
positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]
# build a complete dataset by joining both sets and get the training and testing set for the performance measure of
# model
dataset = positive_dataset + negative_dataset
classifier = nltk.NaiveBayesClassifier.train(dataset)
# Custom testing example
custom_tweet = "predictable with no fun "
custom_tokens = nltk.word_tokenize(custom_tweet)
prediction = classifier.classify(dict([token, True] for token in custom_tokens))
print(custom_tweet + ": ", prediction)
