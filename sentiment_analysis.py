#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import nltk
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy import stats
from nltk.corpus import stopwords
from textblob import TextBlob


# In[2]:


#load cleaned reviews and listings files

reviews_clean = pd.read_csv("reviews_clean.csv")
listings_clean = pd.read_csv("listings_clean.csv")


# In[3]:


#print first few rows of clean reviews file
reviews_clean.head(5)


# In[4]:


#print dataset shape
reviews_clean.shape


# ## WordCloud

# In[5]:


#remove stopwords
stopwords = set(STOPWORDS)
text = " ".join(review for review in reviews_clean.comments)
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=1000, height=500).generate(text)

#save to file
#wordcloud.to_file("review_comments.png")

#plot wordcloud
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Note: I tried to clean these files in the cleaning/EDA stage, but I kept getting an error when I didn't clean the review comments in this .py file. 

# In[6]:


#remove punctuation from reviews

reviews_clean['comments_nopunc'] = reviews_clean['comments'].str.replace('[{}]'.format(string.punctuation), '')
reviews_clean.head()


# In[7]:


#download nltk words to remove non-English from comments

nltk.download('words')


# In[8]:


#remove non-English comments

words = set(nltk.corpus.words.words())

reviews_clean['comments_final'] = [" ".join(w for w in nltk.wordpunct_tokenize(x)
                                             if w.lower() in words or not w.isalpha())
                                                for x in reviews_clean['comments_nopunc']]
reviews_clean.head()


# ## Sentiment Analysis with TextBlob

# In[9]:


#calculate polarity

#define detect polarity function to iterate TextBlob across comments_final column
def detect_pol(comments_final):
    return TextBlob(comments_final).sentiment.polarity

#apply function to clean file and add polarity value to dataset
reviews_clean['polarity'] = reviews_clean.comments_final.apply(detect_pol)
reviews_clean.head(5)


# In[10]:


#calculate subjectivity
def detect_sub(comments_final):
    return TextBlob(comments_final).sentiment.subjectivity

#apply function to reviews file and add subjectivity value to dataset
reviews_clean['subjectivity'] = reviews_clean.comments_final.apply(detect_sub)
reviews_clean.head(5)


# In[11]:


#print comments with lowest polarity
reviews_clean[reviews_clean.polarity == -1].comments_final.head()


# In[12]:


#print comments with highest polarity
reviews_clean[reviews_clean.polarity == 1].comments_final.head()


# In[13]:


#print comments with lowest subjectivity
reviews_clean[reviews_clean.subjectivity == 0].comments_final.head()


# In[14]:


#print comments with highest subjectivity
reviews_clean[reviews_clean.subjectivity == 1].comments_final.head()


# In[15]:


#plot histogram of polarity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(reviews_clean.polarity, num_bins, alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Polarity for Review Comments')
plt.show();


# In[16]:


#plot histogram of subjectivity
num_bins = 30
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(reviews_clean.subjectivity, num_bins, alpha=0.5)
plt.xlabel('Subjectivity')
plt.ylabel('Count')
plt.title('Histogram of Subjectivity for Review Comments')
plt.show();


# ## Join Sentiment to Listings Dataset

# In[17]:


##group reviews by listing id
###average polarity and subjectivity values
group_reviews = reviews_clean.groupby("listing_id")
mean_reviews = group_reviews.mean()
mean_reviews = mean_reviews.reset_index()
mean_reviews = mean_reviews.drop(['id', 'reviewer_id'], axis=1)
mean_reviews.head(5)


# In[18]:


mean_reviews.shape


# In[19]:


#join sentiment values from reviews to listings
listings_sentiment = pd.merge(listings_clean, mean_reviews, left_on='id', right_on='listing_id', how='inner')


# In[20]:


#print first few rows to verify sentiment values in listings
listings_sentiment.head(5)


# In[21]:


#print last few rows to verify sentiment values
listings_sentiment.tail(5)


# In[22]:


listings_sentiment.shape


# In[23]:


#drop duplicate listing id value from reviews
listings_sentiment = listings_sentiment.drop(['listing_id'], axis=1)


# In[24]:


listings_sentiment.shape


# In[25]:


#check for null values
listings_sentiment.isnull().sum()


# In[26]:


#plot histogram of polarity
num_bins = 30
plt.figure(figsize=(10,5))
n, bins, patches = plt.hist(listings_sentiment.polarity, num_bins, alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of Polarity')

#plt.savefig("polarity.png", bbox_inches='tight')

plt.show();


# In[27]:


#plot histogram of subjectivity
num_bins = 30
plt.figure(figsize=(10,5))
n, bins, patches = plt.hist(listings_sentiment.subjectivity, num_bins, alpha=0.5)
plt.xlabel('Subjectivity')
plt.ylabel('Count')
plt.title('Histogram of Subjectivity')

#plt.savefig("subjectivity.png", bbox_inches='tight')

plt.show();


# In[28]:


fig, ax = plt.subplots(figsize=(10,5))
x = listings_sentiment['bedrooms']
y = listings_sentiment['polarity']

ax.set_xlabel('Bedrooms')
ax.set_ylabel('Polarity')
ax.set_title('Polarity vs Bedrooms')

ax.scatter(x,y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")
#plt.savefig("polarity_bed.png", bbox_inches='tight')
plt.show()


# In[29]:


fig, ax = plt.subplots(figsize=(10,5))
x = listings_sentiment['price']
y = listings_sentiment['polarity']

ax.set_xlabel('Price')
ax.set_ylabel('Polarity')
ax.set_title('Polarity vs Price')

ax.scatter(x,y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")
#plt.savefig("polarity_price.png", bbox_inches='tight')
plt.show()


# In[30]:


#plot corr matrix to examine correlation polarity and other features
corr = listings_sentiment.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(20, 12))
cmap = sns.diverging_palette(20, 220, as_cmap=True)
heatm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidth=.5, cbar_kws={"shrink": .5})
heatm.set_title('Correlation Heatmap', fontdict={'fontsize':18})
#plt.savefig("corr_heatmap_sentiment.png", bbox_inches='tight')


# In[31]:


#save updated file
listings_sentiment.to_csv('listings_final.csv', index_label=False)

