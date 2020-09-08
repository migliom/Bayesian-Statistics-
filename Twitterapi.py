import twitter

def buildTestSet(search_keyword, twitter_api):
	tweets_fetched = twitter_api.GetSearch(search_keyword, count = 1)
	print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
	return [{"text":status.text, "label":None} for status in tweets_fetched]

def main():
	twitter_api = twitter.Api(consumer_key='rOofNnlVhYk858RZobDprvfo7',
                        consumer_secret='B81nKhT4whIe6B1sDnJxZppkpgoAuNxyOuGG3UNwbydTPwiGGx',
                        access_token_key='2787664858-NCDTntmKlJL08qrjqZD8dxsOJolpH4QqS6Y1w2q',
                        access_token_secret='0MpBmV0hEFBnAYyrW2y1a8Kzl8gbRoNQfRGtJxPyLq6WT')

	testSet = buildTestSet("Tesla", twitter_api)
	corpusFile = "/Users/matteomiglio/Documents/Python/Bayesian Statistics/corpus.csv"
	tweetDataFile = "/Users/matteomiglio/Documents/Python/Bayesian Statistics/tweetFile.csv"

	trainingData = buildTrainingSet(corpusFile, tweetDataFile, twitter_api)
	return testSet, trainingData
def buildTrainingSet(corpusFile, tweetFile, twitter_api):
	import csv
	import time

	corpus = []
	with open(corpusFile, 'rt') as csvfile:
		readLines  = csv.reader(csvfile,delimiter=',', quotechar="\"")
		for row in readLines:
			corpus.append({"tweet_id":row[2], "label":row[1], "topic":row[0]})
#-------------------------------------------------------------------------------#
	refresh_rate = 180
	sleep_timer = 900 / 180

	trainingSet = []
	x = 1
	for tweet in corpus:
		try:
			print(tweet)
			status = twitter_api.GetStatus(tweet["tweet_id"])
			tweet["text"] = status.text
			trainingSet.append(tweet)
			time.sleep(sleep_timer)
			x +=1
			if x > 10:
				break
		except:
			continue
#-------------------------------------------------------------------------------#
	with open(tweetFile,'wt') as csvfile:
		writeLines = csv.writer(csvfile,delimiter=',',quotechar="\"")
		for tweet in trainingSet:
			try:
				writeLines.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
			except Exception as e:
				print(e)
	return trainingSet
#-------------------------------------------------------------------------------#
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

class tweetCleanup:
	def __init__(self):
		self.stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

	def processTweets(self, list_of_tweets):
		processedTweets=[]
		for tweet in list_of_tweets:
			processedTweets.append((self.cleanTweet(tweet["text"]),tweet["label"]))
		return processedTweets

	def cleanTweet(self, tweet):
		tweet = tweet.lower() # convert text to lower-case
		tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
		tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
		tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
		tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
		return [word for word in tweet if word not in self.stopwords]

#-------------------------------------------------------------------------------#
import nltk

def find_nGrams(preprocessedTrainingData):
    all_words = []

    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()

    return word_features

# ------------------------------------------------------------------------

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features

if __name__ == "__main__":
	testData, trainData = main()

	tweetProcessor = tweetCleanup()
	preprocessedTrainingSet = tweetProcessor.processTweets(trainData)
	preprocessedTestSet = tweetProcessor.processTweets(testData)

	word_features = find_nGrams(preprocessedTrainingSet)
	trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)

	# ------------------------------------------------------------------------

	NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

	# ------------------------------------------------------------------------

	NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]

	if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
		print("Tweet is Showing Positive Sentiment Towards Company")
		print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
	else:
		print("Tweet is Showing Positive Sentiment Towards Company")
		print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
