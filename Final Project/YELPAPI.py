# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:26:20 2015

@author: angelatruong

Convert the Yelp Dataset Challenge dataset from json format to csv.
For more information on the Yelp Dataset Challenge please visit http://yelp.com/dataset_challenge
"""

##################### FEATURE SELECTION AND COMBINING DATA ####################
#Had to add this code due to ascii UnicodeEncodeError
import sys
reload(sys)
sys.setdefaultencoding('UTF8')

import json #yelp challenge dataset was provided as json files
import pandas as pd #pandas for dataframe
import numpy as np
randn = np.random.randn


#REVIEWS DATA
#Read json file from yelp challenge dataset
review = []
with open('yelp_academic_dataset_review.json') as f:
    for line in f:
        review.append(json.loads(line))


#looking at json file content
review
type(review)
len(review)
review[0]
review[-1]
review[0]['business_id']


#Convert json file into a dataframe
df = DataFrame(review)


#Convert and save dataframe into a csv file
df.to_csv('review_yelp.csv', index=False)


#getting 200,000 random sample from the large dataframe
import random
review_yelp = pd.read_csv('review_yelp.csv')

def some(review_yelp,n):
    return review_yelp.ix[np.random.random_integers(0, len(review_yelp),n)]

sample = some(review_yelp,200000)

sample.to_csv('yelp_sample_review.csv', index = False)


#read into new smaller sample dataframe for easier analysis
yelp_sample_review = pd.read_csv('yelp_sample_review.csv')


yelp_sample_review.head()           
yelp_sample_review.tail()           
yelp_sample_review.describe()       
yelp_sample_review.info()           
yelp_sample_review.columns        
yelp_sample_review.shape   


from textblob import TextBlob, Word


#function to detect if a review is in English or not
def isEnglish(sentence):
    try:
        blob = TextBlob(sentence)
        return blob.detect_language() == 'en'
    except:
        return False

#create a binary column if review is English
yelp_sample_review['English'] = yelp_sample_review['text'].map(isEnglish)


yelp_sample_review['English']


# english rows
yelp_sample_review[yelp_sample_review['English'] == True]

translate = yelp_sample_review[yelp_sample_review['English'] == True]


translate.to_csv('all_english.csv',index = False)

english = pd.read_csv('all_english.csv')

english.shape
english.head()
#1,394 of the reviews are not in English and are dropped...



#BUSINESS DATA
business_yelp = pd.read_csv('business_yelp.csv')

business_yelp.shape
business_yelp.head()
business_yelp.tail()

#Hone in on which city had the most data in the Business Dataframe which was VEGAS
#Decided to focus on Vegas since my computer takes long to process for the amount of reviews
business_yelp.city.value_counts()

vegas_lookup = business_yelp[['city','business_id']][business_yelp.city == 'Las Vegas']

vegas_lookup.to_csv('vegas_lookup.csv', index = False)

vdf = pd.read_csv('vegas_lookup.csv', index_col='business_id')

masterdata = english.join(vdf, on='business_id',how = 'left')

masterdataone = masterdata.dropna()



#USER DATA
user_yelp = pd.read_csv('user_yelp.csv')

#Filtering for creditable users who wrote over 100 reviews and are Elites for at least 5 years
def reviewsCategory(number):
    if number >= 100:
        return 'Creditable'
    elif number >= 50:
        return 'Okay'
    return 'Bad'

user_yelp['Useful'] = user_yelp['review_count'].map(reviewsCategory)
user_yelp['Useful'].value_counts()


def elite_5_years(elite):
    return len(elite) > 25

user_yelp['elite_5_years'] = user_yelp['elite'].map(elite_5_years)

user_yelp['cool_people'] = (user_yelp['Useful'] == 'Creditable') & user_yelp['elite_5_years']

user_yelp.head(20)

user_yelp['cool_people'].value_counts()

z = user_yelp[user_yelp['cool_people'] == True]
del z['friends'] #created a mess in the dataset so was dropped. No use for feature.
z.to_csv('creditable_users.csv', index=False)

yelpers = pd.read_csv('creditable_users.csv')

yelpers.info()
yelpers.Useful.value_counts()

my_cols = ['user_id','cool_people']
vlookupyelpers = yelpers[my_cols]
vlookupyelpers.to_csv('coolyelpers.csv', index=False)
coolyelpers = pd.read_csv('coolyelpers.csv', index_col = 'user_id')
masterdatatwo = masterdataone.join(coolyelpers, on='user_id',how = 'left')
masterdatathree = masterdatatwo.dropna()

#Create actual database use for Natural Language Processing for reviews from
# Las Vegas, creditable users, and are in English
masterdatathree.to_csv('yelpdata.csv', index = False)







################## DATA ANALYSIS ##################
yelpdata = pd.read_csv('yelpdata.csv')

yelpdata.head(5)
yelpdata.columns
yelpdata.shape


######### MOST COMMON POSITIVE AND NEGATIVE WORDS ###########
#Corpus of 'text' reviews
import nltk
import re

yelp_sentence = [nltk.word_tokenize(sentence) for sentence in yelpdata['text'] if re.search('^[a-zA-Z]+', sentence)]

from itertools import chain
wordy = list(chain.from_iterable(yelp_sentence))

neg_pos = pd.read_csv('dictionary.csv')

negposdata = [(w, str(neg_pos[neg_pos.Word==w]['Positive-or-Negative'].iloc[0])) for w in wordy if neg_pos[neg_pos.Word==w]['Positive-or-Negative'].count()]

posdata = [line[0] for line in negposdata if line[1] == 'pos']
pd = str(' '.join(posdata))

negdata = [line[0] for line in negposdata if line[1] == 'neg']
nd = str(' '.join(negdata))


from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#Postive WordCloud
pwc = WordCloud(background_color="white", max_words=2000, 
               stopwords=STOPWORDS.add("said"))
# generate word cloud
pwc.generate(pd)

# show
plt.imshow(pwc)
plt.axis("off")
plt.figure()
plt.axis("off")
plt.show()


#Negative WordCloud
nwc = WordCloud(background_color="black", max_words=2000, 
               stopwords=STOPWORDS.add("said"))
# generate word cloud
nwc.generate(nd)

# show
plt.imshow(nwc)
plt.axis("off")
plt.figure()
plt.axis("off")
plt.show()



############## VISUALIZE DATA ###############
yelpdata.stars.hist(bins=20)
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.show()


############# PREDICTIVE MODEL ##############
#Sentiment Polarity: Shows how negative a phrase is. Scale -1 to 1
from textblob import TextBlob, Word

def yelpblober(yelperblob):
    yelpblob = TextBlob(yelperblob)
    return yelpblob.sentiment.polarity

yelpdata['polarity'] = yelper_review.map(yelpblober)
#Not the best method of predicting star rating.



#Naive Bayes 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(yelpdata.text, yelpdata.stars, random_state=1)
X_train.shape
X_test.shape

from sklearn.feature_extraction.text import \
CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import naive_bayes

#TFidfVectorizer      Term Frequency - Inverse Document Frequency
tfidfyelp = TfidfVectorizer(stop_words='english', ngram_range = (1,3), max_features = 50000)
tfidfyelp.fit(yelpdata.text)
tfidf_dtm = tfidfyelp.fit_transform(X_train)
featuresyelp = tfidfyelp.get_feature_names()

tnb = MultinomialNB()
tnb.fit(tfidf_dtm, y_train)

y_predstwo = tnb.predict(tfidfyelp.transform(X_test))

len(y_test)
len(y_predstwo)

float((y_predstwo == y_test).sum()) / len(y_test)

metrics.accuracy_score(y_test, y_predstwo)
metrics.confusion_matrix(y_test, y_predstwo)



##CountVectorizer
yelpvect = CountVectorizer(stop_words = 'english', ngram_range = (1,3), max_features = 50000)
yelpvect.fit(yelpdata.text)
yelp_all_features = yelpvect.get_feature_names()

yelp_all_features

# transform testing data into a document-term matrix
yelptest_dtm = yelpvect.fit_transform(X_train)
yelptest_dtm


from sklearn.naive_bayes import MultinomialNB
from sklearn import naive_bayes
nb = MultinomialNB()
nb.fit(yelptest_dtm, y_train)

y_preds = nb.predict(yelpvect.transform(X_test))

len(y_test)
len(y_preds)

float((y_preds == y_test).sum()) / len(y_test)

