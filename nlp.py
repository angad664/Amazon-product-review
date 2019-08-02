# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:43:49 2019

@author: angadsingh
"""
# importing libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string


#importting the file
df = pd.read_excel('amazon_oneplus7.xlsx', encoding='ISO-8859-1', names=['review','type'] )

# data preprocessing
df.head()
df.describe()
df.isnull().sum()

df['review'] = [word.lower() for word in df['review']]
review = df['review']


#tokenization
from nltk.tokenize import word_tokenize, sent_tokenize
rev_token = word_tokenize(review)


df['sen_length'] = df['review'].apply(len)



review = review.str.replace("[^a-zA-Z#]", " ")

def message_text_processing (mess):
    no_punctuation = [char for char in mess if char not in string.punctuation ]
    no_punctuation = ''.join(no_punctuation)
    no_punctuation
    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]

df['review'].head(5).apply(message_text_processing)

#
from sklearn.feature_extraction.text import CountVectorizer

bag_words = CountVectorizer().fit(df['review'])
print(len(bag_words.vocabulary_))
message_words = bag_words.transform(df['review'])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_trans = TfidfTransformer().fit(message_words)
message_tfidf = tfidf_trans.transform(message_words)
print(message_tfidf.shape)
print(message_tfidf)
# modeling
from sklearn.naive_bayes import MultinomialNB

rev_detect = MultinomialNB().fit( message_tfidf, df['type'])

predicted = rev_detect.predict(message_tfidf)
predicted

expected = df['type']

from sklearn import metrics
from sklearn.metrics import confusion_matrix
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# data visualization
import matplotlib.pyplot as plt
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
stopwords = set(stopwords.words('english'))

# making wordcloud
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40,
                          random_state=42
                         ).generate(str(df['review']))

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
