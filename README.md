# Denver Airbnb Data Analysis Project
This is a practicum project for the MSDS program at Regis University. The original datasets were obtained from http://insideairbnb.com/.

![Denver city skyline photo by Erick Todd from Pexels](https://github.com/tsgruman/regis-practicum-denver-airbnb/blob/assets/pexels-erick-todd-6034694.jpg)
*Denver city skyline photo by Erick Todd from Pexels*

# Introduction
When it comes to vacationing, Airbnb has become synonymous with hotels. The company that was founded in 2007 in San Francisco by two people who opened their home to guests has grown to over 7 million listings in over 220 cities worldwide. The service offers short-term and long-term rentals, ranging from private rooms to whole homes. 

However, not all Airbnb listings are created equally. This project will analyze Denver Airbnb listings to determine what makes an ideal Airbnb listing for Denver visitors. This will be accomplished through sentiment analysis and cluster analysis. First, sentiment analysis using TextBlob will be applied to review comments left by visitors. Second, various clustering techniques will be conducted to cluster Denver listings. The goal is to identify clusters with high polarity values and determine which features contribute to those values.

# Tools
This project was completed using Jupyter Notebook and Python. 

Libraries used include:
* pandas - data manipulation
* matplotlib - plotting
* numpy - arrays
* seaborn - plotting
* datetime - date and time manipulation
* nltk - Natural Language Processing
* string - string datatype manipulation
* wordcloud - generating wordclouds
* textblob
* scipy - mathematical functions
  * scipy.cluster.hierarchy
  * scipy.cluster.hierarchy import dendrogram
* sklearn - variety of machine learning tools
  * preprocessing
  * sklearn.preprocessing import StandardScaler
  * sklearn.cluster import KMeans
  * sklearn.cluster import DBSCAN
  * sklearn.cluster import AgglomerativeClustering
  * sklearn.metrics import silhouette_score
  * sklearn.decomposition import PCA
  * sklearn.neighbors import NearestNeighbors
  * sklearn.feature_selection import SelectKBest
  * sklearn.feature_selection import f_classif

# Data
The datasets were downloaded from InsideAirbnb.com. The Denver listings.csv and reviews.csv files last updated in December 2020 were used. 

## Listings Dataset
The listings.csv file is originally comprised of 3500+ rows and 74 features. After data cleaning to remove NA values, drop columns that were not pertinent to the project, and convert data to the correct datatypes, the cleaned file consisted of 2078 rows and 32 features. 

Features consist of:
* id
* name
* description
* host_id
* host_name
* host_since
* host_location
* host_response_time
* host_response_rate
* host_is_superhost
* host_neighbourhood
* host_listings_count
* host_verifications
* host_has_profile_pic
* host_identity_verified
* latitude
* longitude
* property_type
* room_type
* accommodates
* bathrooms
* bedrooms
* beds
* amenities
* price
* minimum_nights_avg_ntm
* maximum_nights_avg_ntm
* number_of_reviews
* review_scores_rating
* instant_bookable
* reviews_per_month

For cluster analysis and following sentiment analysis, listings are further reduced to 17 columns, which includes the addition of the newly created polarity and subjectivity columns from sentiment analysis.

## Reviews Dataset
The reviews.csv file originally consists of 183,442 rows and 6 columns. Very little data cleaning is needed for this file until the sentiment analysis segment, as I only need the comments column. However, after removing rows with null values, the data is reduced to 183,316 rows.

Features consist of:
* listing_id
* id
* date
* reviewer_id
* reviewer_name
* comments

# Sentiment Analysis
## Data cleaning

This consisted of removing punctuation and using the nltk 'words' library to remove non-English words from the comments. Removing these features reduces the change of affecting the sentiment analysis scores.

## TextBlob
[TextBlob](https://textblob.readthedocs.io/en/dev/) is a powerful and simple-to-use library for text processing. For this project, I used TextBlob for sentiment analysis to calculate polarity and subjectivity scores of comments. 

* Polarity = quantification of positive, neutral, or negative sentiment, range from -1 (negative) to 1 (positive)
* Subjectivity = quantification of degree of opinion vs. objective facts, range from 0 (objective) to 1 (subjective)
