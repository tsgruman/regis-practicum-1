#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import nltk
import os
import string
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy import stats
from nltk.corpus import stopwords


# In[2]:


#load cleaned reviews and listings files

reviews_clean = pd.read_csv("reviews_clean.csv")
listings_clean = pd.read_csv("listings_clean.csv")


# In[3]:


#wordcloud
stopwords = set(STOPWORDS)
text = " ".join(review for review in reviews_clean.comments)
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=1000, height=500).generate(text)

#save to file
wordcloud.to_file("review_comments.png")

#plot wordcloud
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[4]:


#remove punctuation
reviews_clean['comments_clean'] = reviews_clean['comments'].str.replace('[{}]'.format(string.punctuation), '')
reviews_clean.head()


# In[5]:


#download nltk words
nltk.download('words')


# In[6]:


#remove non-English comments

words = set(nltk.corpus.words.words())

reviews_clean['comments_final'] = [" ".join(w for w in nltk.wordpunct_tokenize(x)
                                             if w.lower() in words or not w.isalpha())
                                                for x in reviews_clean['comments_clean']]
reviews_clean.head()


# In[7]:


#sentiment analysis with TextBlob
#https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis

#calculate polarity

from textblob import TextBlob

def detect_pol(comments_final):
    return TextBlob(comments_final).sentiment.polarity

#add polarity value to review file
reviews_clean['polarity'] = reviews_clean.comments_final.apply(detect_pol)
reviews_clean.head(5)


# In[9]:


#calculate subjectivity
def detect_sub(comments_final):
    return TextBlob(comments_final).sentiment.subjectivity

#add subjectivity value to review file
reviews_clean['subjectivity'] = reviews_clean.comments_final.apply(detect_sub)
reviews_clean.head(5)


# In[16]:


#print comments with lowest polarity
reviews_clean[reviews_clean.polarity == -1].comments_final.head()


# In[17]:


#print comments with highest polarity
reviews_clean[reviews_clean.polarity == 1].comments_final.head()


# In[18]:


#print comments with lowest subjectivity
reviews_clean[reviews_clean.subjectivity == 0].comments_final.head()


# In[19]:


#print comments with highest subjectivity
reviews_clean[reviews_clean.subjectivity == 1].comments_final.head()


# In[14]:


#plot histogram of polarity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(reviews_clean.polarity, num_bins, alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Polarity for Review Comments')
plt.show();


# In[15]:


#plot histogram of subjectivity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(reviews_clean.subjectivity, num_bins, alpha=0.5)
plt.xlabel('Subjectivity')
plt.ylabel('Count')
plt.title('Histogram of Subjectivity for Review Comments')
plt.show();


# In[20]:


#prep table to join to listings dataset
##group reviews by listing id
###average polarity and subjectivity values
group_reviews = reviews_clean.groupby("listing_id")
mean_reviews = group_reviews.mean()
mean_reviews = mean_reviews.reset_index()
mean_reviews = mean_reviews.drop(['id', 'reviewer_id'], axis=1)
mean_reviews.head(10)


# In[21]:


mean_reviews.shape


# In[22]:


#join sentiment values from reviews to listings
listings_sentiment = pd.merge(listings_clean, mean_reviews, left_on='id', right_on='listing_id', how='inner')


# In[23]:


listings_sentiment.head(5)


# In[24]:


listings_sentiment.tail(5)


# In[25]:


listings_sentiment.shape


# In[26]:


#drop duplicate listing id value from reviews
listings_sentiment = listings_sentiment.drop(['listing_id'], axis=1)


# In[27]:


listings_sentiment.shape


# In[28]:


#check for null values
listings_sentiment.isnull().sum()


# In[29]:


#plot histogram of polarity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(listings_sentiment.polarity, num_bins, alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Polarity')

plt.savefig("polarity.png", bbox_inches='tight')

plt.show();


# In[30]:


#plot histogram of subjectivity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(listings_sentiment.subjectivity, num_bins, alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Subjectivity')
plt.show();

