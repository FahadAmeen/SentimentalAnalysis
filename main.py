# import nltk
# nltk.download('twitter_samples')
# nltk.download('punkt')
import random
import re
import string

from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.corpus import twitter_samples, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# def get_all_words(cleaned_tokens_list):
#     for tokens in cleaned_tokens_list:
#         for token in tokens:
#             yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


# downloaded the twitter dataset
# get the positive tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
# get the negative tweets
negative_tweets = twitter_samples.strings('negative_tweets.json')
# words like about, the, will
stop_words = stopwords.words('english')
# tokenize the words in files
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
# cleaning of the data
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []
# appended all the words to their arrays after remiving the noise in them
# remove noise is a function in the code
for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

# all_pos_words = get_all_words(positive_cleaned_tokens_list)
#
# freq_dist_pos = FreqDist(all_pos_words)
# print(freq_dist_pos.most_common(10))

# converted the words in datatype of dictionary as required for the classifier
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
# assigned positive to all positive sentiments and negative to all negative sentiments
positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]
# build a complete dataset by joining both sets and get the training and testing set for the performance measure of
# model
dataset = positive_dataset + negative_dataset

random.shuffle(dataset)
# divided in ratio 70:30
# from 0-7000 training set
train_data = dataset[:7000]
# from 7000-onwards for testing set
test_data = dataset[7000:]
# used the NaiveBayes classifier and trained it with training set
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

# print(classifier.show_most_informative_features(10))
# Custom testing example
custom_tweet = "I ordered just once from Subway, they screwed up, never used the app again."

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
