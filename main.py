# import nltk
# nltk.download('twitter_samples')
# nltk.download('punkt')
import random
import re
import string

from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier
from nltk.corpus import twitter_samples, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag

import xlsxwriter


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []
    # removed all the hyperlinks in the tweets
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        # In tagging when a string is noun NN as Tag then it will be 'n', if verb then 'v' and if no one 'a'
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


def get_tweets_for_model(cleaned_tokens_list):
    # converted to dictionary data type
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def get_predicted_results(classifier, test_data):
    predicted_results = []
    for tdata in test_data:
        for tweet in tdata:
            prediction = classifier.classify(dict([token, True] for token in tweet))
            predicted_results.append(prediction)
    return predicted_results


def write_in_excelsheet():
    workbook = xlsxwriter.Workbook('SentimentAnalysis.xlsx')
    worksheet = workbook.add_worksheet('Twitter dataset Predictions')
    cell_format = workbook.add_format({'bold': True, 'font_color': 'red'})
    cell_format_sum = workbook.add_format({'bold': True, 'font_color': 'blue'})
    cell_format_accuracy = workbook.add_format({'bold': True})

    worksheet.write('A1', 'Tweets', cell_format_accuracy)
    worksheet.write('B1', 'Actual Results', cell_format_accuracy)
    worksheet.write('C1', 'Predicted Results', cell_format_accuracy)
    row = 1
    wrong_predictions = 0
    for i in range(3000):
        col = 0
        worksheet.write(row, col, tweets[i])
        col = 1
        worksheet.write(row, col, actual_test_results[i])
        col = 2
        if actual_test_results[i] != predicted_results[i]:
            wrong_predictions += 1
            worksheet.write(row, col, predicted_results[i], cell_format)
        else:
            worksheet.write(row, col, predicted_results[i])
        row += 1
    col = 2
    row += 1
    worksheet.write(row, col, wrong_predictions, cell_format_sum)
    col = 1
    worksheet.write(row, col, "Accuracy:" + str(accuracy), cell_format_sum)
    workbook.close()


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
accuracy = classify.accuracy(classifier, test_data)
# print("Accuracy is:", accuracy)
predicted_results = get_predicted_results(classifier, test_data)
# print("Predicted results", predicted_results)

tweets = []
actual_test_results = []
# joined the sentences
for tweet in test_data:
    tweet_sentence = ' '.join([word for word in tweet[0]])
    tweets.append(tweet_sentence)
    actual_test_results.append(tweet[1])

write_in_excelsheet()
# Custom testing example
custom_tweet = "I ordered just once from Subway, they screwed up, never used the app again."
custom_tokens = remove_noise(word_tokenize(custom_tweet))
print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
