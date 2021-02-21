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


# In[2]:


#set window options
pd.set_option('display.max_columns', 75)
pd.set_option('display.max_rows', 75)


# In[3]:


listings = pd.read_csv("listings.csv")
reviews = pd.read_csv("reviews.csv")


# In[4]:


listings.head(5)


# In[5]:


listings.tail(5)


# In[6]:


listings.shape


# In[7]:


listings.dtypes


# In[8]:


listings.count()


# In[9]:


#show how many variables missing values
listings.isnull().sum()


# In[10]:


listings['bathrooms'] = listings['bathrooms_text'].str.extract("(\d*\.?\d+)")


# In[11]:


listings['bathrooms'] = listings['bathrooms'].astype(float)


# In[12]:


listings = listings.drop(['bathrooms_text'], axis=1)


# In[13]:


listings['price'] = listings['price'].str.extract("(\d*\.?\d+)")


# In[14]:


listings['price'] = listings['price'].astype(float)


# In[15]:


listings.head(5)


# In[16]:


#remove unnecessary columns
listings = listings.drop(['listing_url', 'scrape_id', 'last_scraped', 'picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped', 'first_review', 'last_review', 'neighbourhood_group_cleansed', 'calendar_updated', 'license', 'neighborhood_overview', 'host_about', 'neighbourhood', 'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365'], axis=1)


# In[17]:


listings.shape


# In[64]:


#remove na
listings_clean = listings.dropna()
listings_clean.to_csv('listings_clean.csv', index_label=False)


# In[19]:


listings_clean.shape


# In[20]:


listings_clean.count()


# In[21]:


listings_clean.head(20)


# In[22]:


listings_clean.describe()


# In[23]:


#create dataframe of numerical data
listings_num = listings_clean.select_dtypes(np.number)


# In[24]:


listings_num.dtypes


# In[25]:


listings_num.shape


# In[26]:


listings_clean.host_location.value_counts()


# In[27]:


#boxplots - detect and list outliers
sns.boxplot(x=listings_clean['price'])
plt.savefig("boxplot_price.png", bbox_inches='tight')


# In[28]:


outliers = []
def detect_outliers(list_out):
    threshold = 3
    mean1 = np.mean(list_out)
    std1 = np.std(list_out)
    
    for y in list_out:
        z_score = (y - mean1)/std1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outliers_price = detect_outliers(listings_num['price'])
print(outliers_price)


# In[29]:


#count of price outliers
print(len(outliers_price))


# In[30]:


sns.boxplot(x=listings_clean['minimum_nights_avg_ntm'])
plt.savefig("boxplot_min_nights.png", bbox_inches='tight')


# In[31]:


outliers = []
def detect_outliers(list_out):
    threshold = 3
    mean1 = np.mean(list_out)
    std1 = np.std(list_out)
    
    for y in list_out:
        z_score = (y - mean1)/std1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outliers_minnights = detect_outliers(listings_num['minimum_nights_avg_ntm'])
print(outliers_minnights)


# In[32]:


sns.boxplot(x=listings_clean['maximum_nights_avg_ntm'])
plt.savefig("boxplot_max_nights.png", bbox_inches='tight')


# In[33]:


outliers = []
def detect_outliers(list_out):
    threshold = 3
    mean1 = np.mean(list_out)
    std1 = np.std(list_out)
    
    for y in list_out:
        z_score = (y - mean1)/std1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outliers_maxnights = detect_outliers(listings_num['maximum_nights_avg_ntm'])
print(outliers_maxnights)


# In[34]:


sns.boxplot(x=listings_clean['number_of_reviews'])
plt.savefig("boxplot_num_reviews.png", bbox_inches='tight')


# In[35]:


outliers = []
def detect_outliers(list_out):
    threshold = 3
    mean1 = np.mean(list_out)
    std1 = np.std(list_out)
    
    for y in list_out:
        z_score = (y - mean1)/std1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outliers_numreviews = detect_outliers(listings_num['number_of_reviews'])
print(outliers_numreviews)


# In[36]:


sns.boxplot(x=listings_clean['review_scores_rating'])
plt.savefig("boxplot_ratings.png", bbox_inches='tight')


# In[37]:


outliers = []
def detect_outliers(list_out):
    threshold = 3
    mean1 = np.mean(list_out)
    std1 = np.std(list_out)
    
    for y in list_out:
        z_score = (y - mean1)/std1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outliers_rating = detect_outliers(listings_num['review_scores_rating'])
print(outliers_rating)


# In[38]:


listings_clean.describe()


# In[39]:


listings_clean.hist(column='price')
plt.savefig("price_hist.png", bbox_inches='tight');


# In[40]:


listings_clean.hist(column='review_scores_rating', bins = 15)
plt.savefig("ratings_hist.png", bbox_inches='tight');


# In[41]:


listings_clean.host_is_superhost.value_counts().plot(kind='bar', color="darkcyan")


# In[42]:


listings_clean.host_has_profile_pic.value_counts().plot(kind='bar', color="darkcyan")


# In[43]:


listings_clean.host_identity_verified.value_counts().plot(kind='bar', color="darkcyan")


# In[44]:


#seaborn histogram of property types
plt.figure(figsize=(16,4))
pt = sns.histplot(data=listings_clean, x="property_type")
pt.set(xlabel="Property Type", ylabel="Count")
plt.draw()
pt.set_xticklabels(pt.get_xticklabels(), rotation = 90);


# In[45]:


#pandas histogram of property types
listings_clean.property_type.value_counts().nlargest(40).plot(kind='bar', color="darkcyan", figsize=(15,6))
plt.title("Count of Property Types")
plt.xlabel('Property Type')
plt.savefig("prop_types.png", bbox_inches='tight')


# In[46]:


listings_clean.property_type.value_counts()


# In[47]:


corr = listings_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(20, 12))
cmap = sns.diverging_palette(20, 220, as_cmap=True)
heatm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidth=.5, cbar_kws={"shrink": .5})
heatm.set_title('Correlation Heatmap', fontdict={'fontsize':18});
plt.savefig("corr_heatmap.png", bbox_inches='tight')


# In[48]:


#plot bedrooms and bathrooms with trendline
fig, ax = plt.subplots(figsize=(10,5))
x = listings_clean['bedrooms']
y = listings_clean['bathrooms']

ax.set_xlabel('Bedrooms')
ax.set_ylabel('Bathrooms')
ax.set_title('Bedroom and Bathrooms')

ax.scatter(x,y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")
plt.show()


# In[49]:


fig, ax = plt.subplots(figsize=(10,5))
x = listings_clean['price']
y = listings_clean['bedrooms']
ax.scatter(x, y)
ax.set_xlabel('Price')
ax.set_ylabel('Bedrooms')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")

plt.savefig("price_v_bed.png", bbox_inches='tight')


# In[50]:


fig, ax = plt.subplots(figsize=(10,5))
x = listings_clean['price']
y = listings_clean['bathrooms']
ax.scatter(x, y)
ax.set_xlabel('Price')
ax.set_ylabel('Bathrooms')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")


# In[51]:


#reviews EDA
reviews.shape


# In[52]:


reviews.head(5)


# In[53]:


reviews.tail(5)


# In[54]:


reviews.dtypes


# In[55]:


#check for null values
reviews.isnull().sum()


# In[60]:


#remove null values
reviews = reviews.dropna()
reviews.isnull().sum()


# In[61]:


reviews.shape


# In[65]:


reviews.to_csv('reviews_clean.csv', index_label=False)

