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
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy import stats
from nltk.corpus import stopwords


# In[2]:


reviews_clean = pd.read_csv("reviews_clean.csv")
listings_clean = pd.read_csv("listings_clean.csv")


# In[3]:


#wordcloud
stopwords = set(STOPWORDS)
text = " ".join(review for review in reviews_clean.comments)
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=1000, height=500).generate(text)

#save to file
wordcloud.to_file("review_comments.png")

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[4]:


#sentiment analysis with TextBlob
#https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis

#calculate polarity

from textblob import TextBlob

def detect_pol(comments):
    return TextBlob(comments).sentiment.polarity

reviews_clean['polarity'] = reviews_clean.comments.apply(detect_pol)
reviews_clean.head(5)


# In[5]:


#calculate subjectivity
def detect_sub(comments):
    return TextBlob(comments).sentiment.subjectivity

reviews_clean['subjectivity'] = reviews_clean.comments.apply(detect_sub)
reviews_clean.head(5)


# In[6]:


#comments with lowest polarity
reviews_clean[reviews_clean.polarity == -1].comments.head()


# In[7]:


#comments with highest polarity
reviews_clean[reviews_clean.polarity == 1].comments.head()


# In[8]:


#comments with lowest subjectivity
reviews_clean[reviews_clean.subjectivity == 0].comments.head()


# In[9]:


#comments with highest subjectivity
reviews_clean[reviews_clean.subjectivity == 1].comments.head()


# In[10]:


#histogram of polarity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(reviews_clean.polarity, num_bins, alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Polarity for Review Comments')
plt.show();


# In[11]:


#histogram of subjectivity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(reviews_clean.subjectivity, num_bins, alpha=0.5)
plt.xlabel('Subjectivity')
plt.ylabel('Count')
plt.title('Histogram of Subjectivity for Review Comments')
plt.show();


# In[12]:


group_reviews = reviews_clean.groupby("listing_id")
mean_reviews = group_reviews.mean()
mean_reviews = mean_reviews.reset_index()
mean_reviews = mean_reviews.drop(['id', 'reviewer_id'], axis=1)
mean_reviews.head(10)


# In[13]:


mean_reviews.shape


# In[14]:


listings_sentiment = pd.merge(listings_clean, mean_reviews, left_on='id', right_on='listing_id', how='inner')


# In[15]:


listings_sentiment.head(5)


# In[16]:


listings_sentiment.tail(5)


# In[17]:


listings_sentiment.shape


# In[18]:


listings_sentiment = listings_sentiment.drop(['listing_id'], axis=1)


# In[19]:


listings_sentiment.shape


# In[20]:


listings_sentiment.isnull().sum()


# In[21]:


#histogram of polarity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(listings_sentiment.polarity, num_bins, alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Polarity')

plt.savefig("polarity.png", bbox_inches='tight')

plt.show();


# In[22]:


#histogram of subjectivity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(listings_sentiment.subjectivity, num_bins, alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Subjectivity')
plt.show();

