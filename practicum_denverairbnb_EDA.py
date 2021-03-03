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


# In[2]:


#set window options
pd.set_option('display.max_columns', 75)
pd.set_option('display.max_rows', 75)


# In[3]:


#import listings and reviews files
listings = pd.read_csv("listings.csv")
reviews = pd.read_csv("reviews.csv")


# In[4]:


#print first 5 rows of listings
listings.head(5)


# In[5]:


#print last 5 rows of listings
listings.tail(5)


# ## Data Cleaning

# In[6]:


#examine listings shape - rows and features
listings.shape


# In[7]:


#print listings datatypes
listings.dtypes


# In[8]:


#analyze how many variables contain data - look for discrepancies
listings.count()


# In[9]:


#show how many variables missing values
listings.isnull().sum()


# In[10]:


#extract bathrooms value from string column
listings['bathrooms'] = listings['bathrooms_text'].str.extract("(\d*\.?\d+)")


# In[11]:


#change bathrooms extracted value to float type
listings['bathrooms'] = listings['bathrooms'].astype(float)


# In[12]:


#drop string bathroom column
listings = listings.drop(['bathrooms_text'], axis=1)


# In[13]:


#extract price value from string column
listings['price'] = listings['price'].str.extract("(\d*\.?\d+)")


# In[14]:


#convert datatype to float
listings['price'] = listings['price'].astype(float)


# In[15]:


#print first few rows again to verify data
listings.head(5)


# In[16]:


#plot corr matrix to examine correlation between features
#many are highly correlated, so I can remove many columns
corr = listings.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(20, 12))
cmap = sns.diverging_palette(20, 220, as_cmap=True)
heatm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidth=.5, cbar_kws={"shrink": .5})
heatm.set_title('Correlation Heatmap', fontdict={'fontsize':18})
plt.savefig("corr_heatmap.png", bbox_inches='tight')


# In[17]:


#remove unnecessary columns
listings = listings.drop(['listing_url', 'scrape_id', 'last_scraped', 'picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped', 'host_total_listings_count', 'first_review', 'last_review', 'neighbourhood_group_cleansed', 'calendar_updated', 'license', 'neighborhood_overview', 'host_about', 'neighbourhood', 'neighbourhood_cleansed','minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365','number_of_reviews_ltm', 'number_of_reviews_l30d','calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms'], axis=1)


# In[18]:


#analyze shape of dataset after initial clean-up
listings.shape


# In[19]:


#remove na values
listings_clean = listings.dropna()
listings_clean.to_csv('listings_clean.csv', index_label=False)


# In[20]:


listings_clean.shape


# In[21]:


#print data counts to see any discrepancies
listings_clean.count()


# In[22]:


listings_clean.head(20)


# ## Exploratory Data Analysis: Listings

# In[23]:


#print statistical values for each column
listings_clean.describe()


# In[24]:


#how many hosts actually live in the Denver area?
listings_clean.host_location.value_counts()


# ### Boxplots

# In[25]:


#boxplot of price col
sns.boxplot(x=listings_clean['price'])
plt.savefig("boxplot_price.png", bbox_inches='tight')


# In[26]:


#detect outliers to consider for removal
outliers = []

#define function to detect outliers
def detect_outliers(list_out):
    threshold = 3
    mean1 = np.mean(list_out)
    std1 = np.std(list_out)
    
    for y in list_out:
        z_score = (y - mean1)/std1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

#apply to dataset
outliers_price = detect_outliers(listings_clean['price'])
print(outliers_price)


# In[27]:


#print count of price outliers
print(len(outliers_price))


# In[28]:


#boxplot of minimum nights
sns.boxplot(x=listings_clean['minimum_nights_avg_ntm'])
plt.savefig("boxplot_min_nights.png", bbox_inches='tight')


# In[29]:


#detect outliers in minimum nights to consider for removal
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

outliers_minnights = detect_outliers(listings_clean['minimum_nights_avg_ntm'])
print(outliers_minnights)


# In[30]:


#boxplot of maximinum nights
sns.boxplot(x=listings_clean['maximum_nights_avg_ntm'])
plt.savefig("boxplot_max_nights.png", bbox_inches='tight')


# In[31]:


#detect outliers of maximimum nights
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

outliers_maxnights = detect_outliers(listings_clean['maximum_nights_avg_ntm'])
print(outliers_maxnights)


# In[32]:


#boxplot of number of reviews
sns.boxplot(x=listings_clean['number_of_reviews'])
plt.savefig("boxplot_num_reviews.png", bbox_inches='tight')


# In[33]:


#detect outliers for reviews
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

outliers_numreviews = detect_outliers(listings_clean['number_of_reviews'])
print(outliers_numreviews)


# In[34]:


#boxplot of review scores
sns.boxplot(x=listings_clean['review_scores_rating'])
plt.savefig("boxplot_ratings.png", bbox_inches='tight')


# In[35]:


#detect outliers for ratings
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

outliers_rating = detect_outliers(listings_clean['review_scores_rating'])
print(outliers_rating)


# ### Histograms and Bar Charts

# In[36]:


#plot histogram of price distribution
listings_clean.hist(column='price')
plt.savefig("price_hist.png", bbox_inches='tight');


# In[37]:


#plot histogram of ratings
listings_clean.hist(column='review_scores_rating', bins = 15)
plt.savefig("ratings_hist.png", bbox_inches='tight');


# In[38]:


#plot whether host is superhost
listings_clean.host_is_superhost.value_counts().plot(kind='bar', color="darkcyan")
plt.title("Host is Superhost")


# In[39]:


#plot whether host has profile photo
listings_clean.host_has_profile_pic.value_counts().plot(kind='bar', color="darkcyan")
plt.title("Host Has Profile Picture")


# In[40]:


#plot whether host identity is verified
listings_clean.host_identity_verified.value_counts().plot(kind='bar', color="darkcyan")
plt.title("Host Identity Verified")


# In[41]:


#seaborn histogram of property types - what are most common property types?
plt.figure(figsize=(16,4))
pt = sns.histplot(data=listings_clean, x="property_type")
pt.set(xlabel="Property Type", ylabel="Count")
plt.draw()
pt.set_xticklabels(pt.get_xticklabels(), rotation = 90)
plt.title("Histrogram of Property Types - Seaborn");


# In[42]:


#pandas histogram of property types
listings_clean.property_type.value_counts().nlargest(40).plot(kind='bar', color="darkcyan", figsize=(15,6))
plt.title("Count of Property Types")
plt.xlabel('Property Type')
plt.savefig("prop_types.png", bbox_inches='tight')
plt.title("Histogram of Property Types - Pandas")


# In[43]:


#print counts of property types
listings_clean.property_type.value_counts()


# ### Correlation Matrix

# In[44]:


#plot corr matrix to examine correlation between features
corr = listings_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(20, 12))
cmap = sns.diverging_palette(20, 220, as_cmap=True)
heatm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidth=.5, cbar_kws={"shrink": .5})
heatm.set_title('Correlation Heatmap', fontdict={'fontsize':18})
plt.savefig("corr_heatmap.png", bbox_inches='tight')


# ### Scatterplots

# In[45]:


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
plt.savefig("bed_v_bath.png", bbox_inches='tight')
plt.show()


# In[46]:


fig, ax = plt.subplots(figsize=(10,5))
x = listings_clean['price']
y = listings_clean['bedrooms']
ax.scatter(x, y)
ax.set_xlabel('Price')
ax.set_ylabel('Bedrooms')
ax.set_title('Price and Bedrooms')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")

plt.savefig("price_v_bed.png", bbox_inches='tight')


# In[47]:


fig, ax = plt.subplots(figsize=(10,5))
x = listings_clean['price']
y = listings_clean['bathrooms']
ax.scatter(x, y)
ax.set_xlabel('Price')
ax.set_ylabel('Bathrooms')
ax.set_title('Price and Bathrooms')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")

plt.savefig("price_v_bath.png", bbox_inches='tight')


# In[48]:


fig, ax = plt.subplots(figsize=(10,5))
x = listings_clean['price']
y = listings_clean['bathrooms']
ax.scatter(x, y)
ax.set_xlabel('Price')
ax.set_ylabel('Bathrooms')
ax.set_title('Price and Bathrooms')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")

plt.savefig("price_v_bath.png", bbox_inches='tight')


# ## EDA: Reviews

# In[49]:


#initial shape of the dataset
reviews.shape


# In[50]:


#print first few rows of reviews
reviews.head(5)


# In[51]:


#print last few rows of reviews
reviews.tail(5)


# In[52]:


#print datatypes
reviews.dtypes


# In[53]:


#check for null values
reviews.isnull().sum()


# In[54]:


#remove null values
reviews = reviews.dropna()
reviews.isnull().sum()


# In[55]:


#verify reviews dataset shape after removing NA
reviews.shape


# In[56]:


#save clean file locally for sentiment analysis
reviews.to_csv('reviews_clean.csv', index_label=False)

