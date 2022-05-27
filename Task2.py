import json
import re, jieba, string, emoji
from sklearn.model_selection import train_test_split
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import math
import numpy as np

class Filter:
    def __init__(self, tweetlist, stopwords):
        '''
        :param tweetlist: a list of tweet of the model
        :param stopwords: a list of stopwords
        '''
        self.tweetlist = tweetlist
        self.stopwords = stopwords

    def is_valid_tweet(self, word):
        '''
        :param word: a string
        :return: a string if not in stopwords, else return None
        '''
        try:
            return((word not in self.stopwords) and (word != '&amp;'))
        except Exception as e:
            return None


    def normalize(self,word):
        '''
        :param word: a string
        :return: a clean string without punctuations in lower case
        '''
        if(word.startswith('#') or word.startswith('@') or word.startswith('$')):
            word = word[1:]
        if(word.startswith('T&amp') or word.startswith('http')):
            return None
        s = word.lower()
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if (ch not in exclude))
        return s

    def cleanList(self, tweet):
        '''
        :param tweet: a tweet String
        :return: a tweet String without emoji and being encode and decode to the same coding style
        '''
        text = re.sub(emoji.get_emoji_regexp(), r"", tweet)
        text = text.encode("ascii", errors="ignore").decode()
        return text

    def clean(self):
        '''
        :return: a tweet list and each element is a list of word after cleansing
        '''
        tweet_list = []
        filter = Filter(self.tweetlist, self.stopwords)
        for tweet in self.tweetlist:
            tweet = filter.cleanList(tweet)
            words = jieba.lcut(tweet, cut_all=False, HMM=True)
            word_list = []
            for word in words:
                if (word is not None and filter.is_valid_tweet(word) == True):
                    word = filter.normalize(word)
                    if word is not None and len(word.strip()) > 1 and word not in self.stopwords:
                        word_list.append(word)

            tweet_list.extend(word_list)
        return tweet_list

    def word_dict(self):
        '''
        :return: a dictionary, each key is a word in the tweets and the value is the frequency of the word in the whole model
        '''
        tweet_list = Filter(self.tweetlist, self.stopwords).clean()
        word_dict = {}
        for key in tweet_list:
            word_dict[key] = word_dict.get(key, 0) + 1
        return word_dict

class NewsworthinessScoring:
    def __init__(self, hqTermweight, lqTermweight, threshold):
        self.hqTermweight = hqTermweight ## the dictionary of word frequency in high quality model
        self.lqTermweight = lqTermweight ## the dictionary of word frequency in low quality model
        self.threshold = threshold ## the threshold in CNTD

    def CNTD(self):
        f_h = sum(self.hqTermweight.values())
        f_l = sum(self.lqTermweight.values())
        bgweight = {}
        bgweight.update(self.hqTermweight)
        for key,value in self.lqTermweight.items():
            bgweight[key] = bgweight.get(key, 0) + value
        f_bg = sum(bgweight.values())

        R_hq = {}
        R_lq = {}
        for key, value in self.hqTermweight.items():
            tf_h = value
            tf_bg = bgweight.get(key, 0)
            R_hq[key] = (tf_h * f_bg) / (tf_bg * f_h)
        for key, value in self.lqTermweight.items():
            tf_l = value
            tf_bg = bgweight.get(key, 0)
            R_lq[key] = (tf_l * f_bg) / (tf_bg * f_l)

        S_hq = {} # score for each term in high quality model
        S_lq = {} # score for each term in low quality model
        for key, value in R_hq.items():
            if value >= self.threshold:
                S_hq[key] = value
            else:
                S_hq[key] = 0
        for key, value in R_lq.items():
            if value >= self.threshold:
                S_lq[key] = value
            else:
                S_lq[key] = 0

        return S_hq, S_lq


    def Scoring(self, tweet_list):
        S_hq, S_lq = NewsworthinessScoring.CNTD(self)
        S1, S2 = 0, 0
        for key in tweet_list:
            S1 += S_hq.get(key, 0)
            S2 += S_lq.get(key, 0)
        N = math.log2((1+S1) / (1+S2))
        return N

    def tweetlist_Scoring(self, tweets): ## tweets is a list of tweets
        n = []
        model = NewsworthinessScoring(self.hqTermweight, self.lqTermweight, self.threshold)
        for tweet in tweets:
            tweet_clean = Filter([tweet],stopwords).clean()
            tweet_list = list(set(tweet_clean))
            n.append(model.Scoring(tweet_list))
        return n


## loading stopwords and major words
stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
jieba.load_userdict("./major_term.txt")
## hq
highFileFeb = []
for line1 in open('highFileFeb', 'r', encoding="utf-8"):
    line_json1 = json.loads(line1)
    highFileFeb.append(line_json1["text"])
hq_dict = Filter(highFileFeb,stopwords).word_dict()

## lq
lowFileFeb = []
for line2 in open('lowFileFeb', 'r', encoding="utf-8"):
    line_json2 = json.loads(line2)
    lowFileFeb.append(line_json2["text"])
lq_dict = Filter(lowFileFeb,stopwords).word_dict()

# wordcloud
def wordcloud(dict,filename, background_color='white', width=900, height=600, max_words=100, max_font_size=99, min_font_size=16, random_state=50):
    my_cloud = WordCloud(
        background_color=background_color,
        width=width, height=height,
        max_words=max_words,
        max_font_size=max_font_size, min_font_size=min_font_size,
        random_state=random_state
    ).generate_from_frequencies(dict)
    plt.imshow(my_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename)
    plt.show()

model = NewsworthinessScoring(hq_dict, lq_dict, 1.52)


if __name__ == '__main__':
    # divide the datasets into training and testing datasets
    train_hq, test_hq = train_test_split(highFileFeb, test_size=0.4, random_state=42)
    train_lq, test_lq = train_test_split(lowFileFeb, test_size=0.4, random_state=42)

    train_hq_dict = Filter(train_hq, stopwords).word_dict()
    train_lq_dict = Filter(train_lq, stopwords).word_dict()
    test_hq_dict = Filter(test_hq, stopwords).word_dict()
    test_lq_dict = Filter(test_lq, stopwords).word_dict()
    # print(train_hq_dict)

    # choosing the appropriate threshold
    threshold_train = pd.DataFrame(columns=["threshold", "Recall", "Precison", "F1"])
    j = 0 # the index of threshold_train
    for i in list(np.linspace(1.5, 2, 26)):
        model_train = NewsworthinessScoring(train_hq_dict, train_lq_dict, i)
        n1 = model_train.tweetlist_Scoring(train_hq)
        n2 = model_train.tweetlist_Scoring(train_lq)
        # high quality is T, low quality is F
        actual_T = len(n1)
        actual_F = len(n2)
        # TP is the number of high quality tweets with score larger than 0, TN is the number of low quality tweets with score lower than 0
        TP = np.sum(list(map(lambda x: x > 0, n1)))
        TN = np.sum(list(map(lambda x: x < 0, n2)))
        FN = actual_T - TP
        FP = actual_F - FN
        Recall = round(TP / actual_T, 4)
        Precision = round(TP / (TP + FP), 4)
        F1 = round( 2 * (Recall * Precision) / (Recall + Precision), 4)
        threshold_train.loc[j] = [i, Recall, Precision, F1]
        j += 1

    # choosing the top 5 F1 score threshold values, then choose the top 3 Recall of the values.
    top5_F1 = threshold_train.sort_values(['F1'], ascending=False).head(5)
    top3_Recall = top5_F1.sort_values(['Recall'], ascending=False).head(3)
    print(top3_Recall)
    thresholds = top3_Recall['threshold']


    # compare the quality of models in testing datasets
    threshold_test = pd.DataFrame(columns=["threshold", "Recall", "Precison", "F1"])
    j = 0
    for threshold in thresholds:
        model_test = NewsworthinessScoring(test_hq_dict, test_lq_dict, threshold)
        n1 = model_test.tweetlist_Scoring(test_hq)
        n2 = model_test.tweetlist_Scoring(test_lq)
        # high quality is T, low quality is F
        actual_T = len(n1)
        actual_F = len(n2)
        # TP is the number of high quality tweets with score larger than 0, TN is the number of low quality tweets with score lower than 0
        TP = np.sum(list(map(lambda x: x > 0, n1)))
        TN = np.sum(list(map(lambda x: x < 0, n2)))
        FN = actual_T - TP
        FP = actual_F - FN
        Recall = round(TP / actual_T, 4)
        Precision = round(TP / (TP + FP), 4)
        F1 = round(2 * (Recall * Precision) / (Recall + Precision), 4)
        threshold_test.loc[j] = [threshold, Recall, Precision, F1]
        j += 1
    print(threshold_test)


    file_name1 = "Task2_hq_wordcloud.png"
    wordcloud(hq_dict, file_name1)
    file_name2 = "Task2_lq_wordcloud.png"
    wordcloud(lq_dict, file_name2)
    n = model.tweetlist_Scoring(highFileFeb)
    print(sorted(n))


