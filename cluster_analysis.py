#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#clustering techniques
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

#feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions


# In[2]:


#set display options for max columns
#allows me to see all columns without trunc
pd.set_option('display.max_columns', 75)


# In[3]:


#load final listings file after sentiment analysis
listings = pd.read_csv("listings_final.csv")

#select only numerical datatypes
listings_num = listings.select_dtypes(np.number)

#convert all to type float from int
listings_num['id'] = listings_num['id'].astype(float)
listings_num['host_id'] = listings_num['host_id'].astype(float)
listings_num['accommodates'] = listings_num['accommodates'].astype(float)
listings_num['number_of_reviews'] = listings_num['number_of_reviews'].astype(float)

listings_num.shape


# In[4]:


#plot histograms of numerical data to analyze distribution
listings_num.hist(figsize=(10,10))


# In[5]:


#standardize data to reduce skew
scaler = StandardScaler()
scaled = scaler.fit_transform(listings_num)
standard_cluster = pd.DataFrame(scaled, columns=listings_num.columns)
standard_cluster.head()


# In[6]:


#analyze statistical values of scaled data
standard_cluster.describe()


# In[7]:


#feature selection with ExtraTreesClassifier
from sklearn.feature_selection import f_regression

array = standard_cluster.values
X = array[:,0:16]
Y = array[:,16]

fs = SelectKBest(score_func=f_regression, k=5)
X_new = fs.fit_transform(X, Y)
print(X_new.shape)


# In[8]:


print(X_new[:5])


# Comparing this to dataset, feature selection is recommending minimum nights, number of reviews, rating scores, reviews per month, and polarity.

# In[9]:


#feature selection SelectKBest

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions

array = standard_cluster.values
X = array[:,0:16]
Y = array[:,16]

test = SelectKBest(score_func=f_classif, k=5)
fit = test.fit(X, Y)

set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features[0:5,:])


# Comparing the SelectKBest results to the dataset, recommended features are id, accommodates, bedrooms, number of reviews, and reviews per month.

# In[10]:


#create subset of data based on ExtraTreesClassifier feature selection
cluster_sub = standard_cluster[['minimum_nights_avg_ntm', 'number_of_reviews', 'review_scores_rating', 'reviews_per_month', 'polarity']]
cluster_sub.head()


# In[11]:


#normalize data
n = preprocessing.normalize(listings_num)

normal_cluster = pd.DataFrame(data = n, columns=listings_num.columns)
normal_cluster.head()


# ## Clustering: K-Means

# I have 3 separate datasets to work with now: the raw data (listings_num), standardized data (standard_cluster), and normalized data (normal_cluster). 

# ### K-Means with Raw Data

# In[12]:


#Elbow method for optimal k clusters
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(listings_num)
    distortions.append(kmeanModel.inertia_)


# In[13]:


#plot resulting elbow method plot
plt.figure(figsize=(10,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal k')
plt.show()


# Elbow method shows that 3 is the optimal number of clusters.

# In[14]:


#apply KMeans
kmeans = KMeans(n_clusters = 3).fit(listings_num)
y_means = kmeans.predict(listings_num)

#add labels to original data
labels = pd.DataFrame(kmeans.labels_)

labelListings = pd.concat((listings_num,labels),axis=1)
labelListings = labelListings.rename({0:'labels'}, axis=1)

labelListings[['id', 'bathrooms', 'bedrooms', 'price', 'polarity', 'labels']].head(5)


# In[15]:


#plot pairs 
sns.pairplot(labelListings, hue='labels')


# The pair plots are incredibly difficult to read with the number of variables and plots it produced. I will instead use strip plots to visualize the variables and clusters.

# In[16]:


#plot stripplots
#https://www.kaggle.com/ellecf/visualizing-multidimensional-clusters

f, axes = plt.subplots(4, 5, figsize=(20,25), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings))-1):
    col = labelListings.columns[i]
    if i < 5:
        ax = sns.stripplot(x=labelListings['labels'], y=labelListings[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 5 and i < 10:
        ax = sns.stripplot(x=labelListings['labels'],y=labelListings[col].values, jitter=True, ax=axes[1,(i-10)])
        ax.set_title(col)
    elif i >= 10 and i < 15:
        ax = sns.stripplot(x=labelListings['labels'], y=labelListings[col].values, jitter=True, ax=axes[2, (i-10)])
        ax.set_title(col)
    elif i >= 15:
        ax = sns.stripplot(x=labelListings['labels'], y = labelListings[col].values, jitter=True, ax=axes[3, (i-15)])
        ax.set_title(col)
        
plt.savefig("stripplots_original.png", bbox_inches='tight')


# In[17]:


#plot strip plots with subset data identified by feature selection

#concat labels to subset data
labelListings_sub = pd.concat((cluster_sub,labels),axis=1)
labelListings_sub = labelListings_sub.rename({0:'labels'}, axis=1)

f, axes = plt.subplots(2,3, figsize=(20,10), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_sub))-1):
    col = labelListings_sub.columns[i]
    if i < 3:
        ax = sns.stripplot(x=labelListings_sub['labels'], y=labelListings_sub[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 3:
        ax = sns.stripplot(x=labelListings_sub['labels'],y=labelListings_sub[col].values, jitter=True, ax=axes[1,(i-3)])
        ax.set_title(col)


# ### K-Means with Standardized Data

# In[18]:


stand_scores = [KMeans(n_clusters=i+1).fit(standard_cluster).inertia_
         for i in range(20)]
sns.lineplot(np.arange(2,22), stand_scores)


# The results aren't very definitive, but I will go with an estimated k = 6.

# In[19]:


#perform kmeans with standardized data k=6
kmeans_standard = KMeans(n_clusters=6).fit(standard_cluster)

y_means_stand = kmeans_standard.predict(standard_cluster)


# In[20]:


#add cluster labels to original data and subset data
labels = pd.DataFrame(kmeans_standard.labels_)

labelListings_standard = pd.concat((listings_num,labels),axis=1)
labelListings_standard = labelListings_standard.rename({0:'labels'}, axis=1)

labelListings_standard_sub = pd.concat((cluster_sub,labels),axis=1)
labelListings_standard_sub = labelListings_standard_sub.rename({0:'labels'}, axis=1)

labelListings_standard.head()


# In[21]:


#plot strip plots

f, axes = plt.subplots(4, 5, figsize=(20,25), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_standard))-1):
    col = labelListings_standard.columns[i]
    if i < 5:
        ax = sns.stripplot(x=labelListings_standard['labels'], y=labelListings_standard[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 5 and i < 10:
        ax = sns.stripplot(x=labelListings_standard['labels'],y=labelListings_standard[col].values, jitter=True, ax=axes[1,(i-10)])
        ax.set_title(col)
    elif i >= 10 and i < 15:
        ax = sns.stripplot(x=labelListings_standard['labels'], y=labelListings_standard[col].values, jitter=True, ax=axes[2, (i-10)])
        ax.set_title(col)
    elif i >= 15:
        ax = sns.stripplot(x=labelListings_standard['labels'], y = labelListings_standard[col].values, jitter=True, ax=axes[3, (i-15)])
        ax.set_title(col)


# In[22]:


#repeat plots with subset data

f, axes = plt.subplots(2,3, figsize=(20,10), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_standard_sub))-1):
    col = labelListings_standard_sub.columns[i]
    if i < 3:
        ax = sns.stripplot(x=labelListings_standard_sub['labels'], y=labelListings_standard_sub[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 3:
        ax = sns.stripplot(x=labelListings_standard_sub['labels'],y=labelListings_standard_sub[col].values, jitter=True, ax=axes[1,(i-3)])
        ax.set_title(col)


# ### K-Means with Normalized Data

# In[23]:


norm_scores = [KMeans(n_clusters=i+1).fit(normal_cluster).inertia_
         for i in range(10)]
sns.lineplot(np.arange(2,12), norm_scores)


# The resulting plot gives a clear indication of k=4.

# In[24]:


#perform kmeans with normalized data k=4
kmeans_normal = KMeans(n_clusters=4).fit(normal_cluster)

y_means_norm = kmeans_normal.predict(normal_cluster)


# In[25]:


#add cluster labels to original data and subset data
labels = pd.DataFrame(kmeans_normal.labels_)

labelListings_normal = pd.concat((listings_num,labels),axis=1)
labelListings_normal = labelListings_normal.rename({0:'labels'}, axis=1)

labelListings_normal_sub = pd.concat((cluster_sub,labels),axis=1)
labelListings_normal_sub = labelListings_normal_sub.rename({0:'labels'}, axis=1)

labelListings_normal.head()


# In[26]:


#plot strip plots

f, axes = plt.subplots(4, 5, figsize=(20,25), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_normal))-1):
    col = labelListings_normal.columns[i]
    if i < 5:
        ax = sns.stripplot(x=labelListings_normal['labels'], y=labelListings_normal[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 5 and i < 10:
        ax = sns.stripplot(x=labelListings_normal['labels'],y=labelListings_normal[col].values, jitter=True, ax=axes[1,(i-10)])
        ax.set_title(col)
    elif i >= 10 and i < 15:
        ax = sns.stripplot(x=labelListings_normal['labels'], y=labelListings_normal[col].values, jitter=True, ax=axes[2, (i-10)])
        ax.set_title(col)
    elif i >= 15:
        ax = sns.stripplot(x=labelListings_normal['labels'], y = labelListings_normal[col].values, jitter=True, ax=axes[3, (i-15)])
        ax.set_title(col)
        
#save strip plot of best scored cluster model
plt.savefig("normal_stripplot.png",bbox_inches='tight')


# In[27]:


#repeat plots with subset data

f, axes = plt.subplots(2,3, figsize=(20,10), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_normal_sub))-1):
    col = labelListings_normal_sub.columns[i]
    if i < 3:
        ax = sns.stripplot(x=labelListings_normal_sub['labels'], y=labelListings_normal_sub[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 3:
        ax = sns.stripplot(x=labelListings_normal_sub['labels'],y=labelListings_normal_sub[col].values, jitter=True, ax=axes[1,(i-3)])
        ax.set_title(col)


# ## Clustering: DBSCAN

# In[28]:


#find optimal eps value with NearestNeighbors function
#https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd

neighbors = NearestNeighbors(n_neighbors=30)
neighbors_fit = neighbors.fit(standard_cluster)
distances, indices = neighbors_fit.kneighbors(standard_cluster)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title("Point Distances")


# In[29]:


#run dbscan with eps set at various values according to point distances plot
for eps in [1, 3, 5, 7]:
    print("\neps={}".format(eps))
    min_samples=5
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels=dbscan.fit_predict(standard_cluster)
    print("min_samples= {}".format(min_samples))
    print("Number of clusters: {}".format(len(np.unique(labels))))
    print("Cluster sizes: {}".format(np.bincount(labels + 1)))


# In[30]:


#run dbscan with min_samples at various values
for min_samples in [1, 3, 5, 7, 9]:
    print("\nmin_samples={}".format(min_samples))
    eps=3
    dbscan2 = DBSCAN(min_samples=min_samples, eps=eps)
    labels=dbscan2.fit_predict(standard_cluster)
    print("eps: {}".format(eps))
    print("Number of clusters: {}".format(len(np.unique(labels))))
    print("Cluster sizes: {}".format(np.bincount(labels + 1)))


# The optimal DBSCAN values seem to be eps=3 and min_samples=5.

# In[31]:


#apply DBSCAN
dbscan_cluster = DBSCAN(eps=3, min_samples=5).fit(standard_cluster)


# In[32]:


#add cluster labels to original data and subset data
labels_db = dbscan_cluster.labels_

labelListings_db = listings_num.copy()
labelListings_db["labels"]=labels_db

labelListings_db_sub = cluster_sub
labelListings_db_sub["labels"]=labels_db

labelListings_db.head()


# In[33]:


labelListings_db_sub.head()


# In[34]:


#plot strip plots to full dataset
f, axes = plt.subplots(4, 5, figsize=(20,25), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_db))-1):
    col = labelListings_db.columns[i]
    if i < 5:
        ax = sns.stripplot(x=labelListings_db['labels'], y=labelListings_db[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 5 and i < 10:
        ax = sns.stripplot(x=labelListings_db['labels'],y=labelListings_db[col].values, jitter=True, ax=axes[1,(i-10)])
        ax.set_title(col)
    elif i >= 10 and i < 15:
        ax = sns.stripplot(x=labelListings_db['labels'], y=labelListings_db[col].values, jitter=True, ax=axes[2, (i-10)])
        ax.set_title(col)
    elif i >= 15:
        ax = sns.stripplot(x=labelListings_db['labels'], y = labelListings_db[col].values, jitter=True, ax=axes[3, (i-15)])
        ax.set_title(col)


# In[35]:


#repeat plots with subset data
#labels -1 are identified as noise by DBSCAN

f, axes = plt.subplots(2,3, figsize=(20,10), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_db_sub))-1):
    col = labelListings_db_sub.columns[i]
    if i < 3:
        ax = sns.stripplot(x=labelListings_db_sub['labels'], y=labelListings_db_sub[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 3:
        ax = sns.stripplot(x=labelListings_db_sub['labels'],y=labelListings_db_sub[col].values, jitter=True, ax=axes[1,(i-3)])
        ax.set_title(col)


# ## Clustering: Agglomerative

# In[36]:


#plot dendrogram to visualize number of clusters
plt.figure(figsize=(20, 8))
plt.title('Hierarchy Dendrogram')
dend = sch.dendrogram((sch.linkage(standard_cluster, method='ward')))


# In[37]:


#apply clustering with k=2
agg = AgglomerativeClustering(n_clusters=2).fit(standard_cluster)


# In[38]:


#add cluster labels to original data and subset data
labels_agg = agg.labels_

labelListings_agg = listings_num.copy()
labelListings_agg["labels"]=labels_agg

labelListings_agg_sub = cluster_sub
labelListings_agg_sub["labels"]=labels_agg

labelListings_agg.head()


# In[39]:


#plot strip plots to full dataset
f, axes = plt.subplots(4, 5, figsize=(20,25), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_agg))-1):
    col = labelListings_agg.columns[i]
    if i < 5:
        ax = sns.stripplot(x=labelListings_agg['labels'], y=labelListings_agg[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 5 and i < 10:
        ax = sns.stripplot(x=labelListings_agg['labels'],y=labelListings_agg[col].values, jitter=True, ax=axes[1,(i-10)])
        ax.set_title(col)
    elif i >= 10 and i < 15:
        ax = sns.stripplot(x=labelListings_agg['labels'], y=labelListings_agg[col].values, jitter=True, ax=axes[2, (i-10)])
        ax.set_title(col)
    elif i >= 15:
        ax = sns.stripplot(x=labelListings_agg['labels'], y = labelListings_agg[col].values, jitter=True, ax=axes[3, (i-15)])
        ax.set_title(col)


# In[40]:


#repeat plots with subset data
#labels -1 are identified as noise by DBSCAN

f, axes = plt.subplots(2,3, figsize=(20,10), sharex=False)
f.subplots_adjust(hspace=0.2, wspace=0.7)

for i in range(0,len(list(labelListings_agg_sub))-1):
    col = labelListings_agg_sub.columns[i]
    if i < 3:
        ax = sns.stripplot(x=labelListings_agg_sub['labels'], y=labelListings_agg_sub[col].values, jitter=True, ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 3:
        ax = sns.stripplot(x=labelListings_agg_sub['labels'],y=labelListings_agg_sub[col].values, jitter=True, ax=axes[1,(i-3)])
        ax.set_title(col)


# ## Evaluate Models with Silhouette Scores

# In[41]:


#print silhouette scores to evaluate cluster models

print('kmeans: {}'.format(silhouette_score(listings_num, kmeans.labels_)))
print('Normalized kmeans: {}'.format(silhouette_score(normal_cluster, kmeans_normal.labels_)))
print('Standardized kmeans: {}'.format(silhouette_score(standard_cluster, kmeans_standard.labels_)))
print('DBSCAN: {}'.format(silhouette_score(standard_cluster, dbscan_cluster.labels_)))
print('Agglomerative: {}'.format(silhouette_score(standard_cluster, agg.labels_)))

