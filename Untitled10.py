#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
sns.set_style("whitegrid") # Plotting style
np.random.seed(7) # seeding random number generator


# In[9]:


df=pd.read_csv("https://raw.githubusercontent.com/Arjun-Mota/amazon-product-reviews-sentiment-analysis/master/1429_1.csv")
df.head()


# In[10]:


df.shape


# In[11]:


df.tail()


# In[12]:


df.columns


# In[13]:


data = df.copy()
data


# In[14]:


data.describe()


# In[15]:


data.info()


# In[16]:


data["asins"].unique()


# In[17]:


asins_unique = len(data["asins"].unique())
print("Number of Unique ASINs: " + str(asins_unique))


# In[18]:


## Visualizing the distributions of numerical variables:

data.hist(bins=50, figsize=(20,15))
plt.show()


# In[19]:


## Split the data into Train and Test

from sklearn.model_selection import StratifiedShuffleSplit
print("Before {}".format(len(data)))
dataAfter = data.dropna(subset=["reviews.rating"])
# Removes all NAN in reviews.rating
print("After {}".format(len(dataAfter)))
dataAfter["reviews.rating"] = dataAfter["reviews.rating"].astype(int)

split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in split.split(dataAfter,
                                           dataAfter["reviews.rating"]):
    strat_train = dataAfter.reindex(train_index)
    strat_test = dataAfter.reindex(test_index)


# In[20]:


## We need to see if train and test sets were stratified proportionately in comparison to raw data:

print(len(strat_train))
print(len(strat_test))
print(strat_test["reviews.rating"].value_counts()/len(strat_test))


# In[21]:


#### Data Exploration (Training Set)
reviews = strat_train.copy()
reviews.head()


# In[22]:


print(len(reviews["name"].unique()), len(reviews["asins"].unique()))


# In[23]:


print(reviews.info())


# In[24]:


print(reviews.groupby("asins")["name"].unique())


# In[25]:


## Lets see all the different names for this product that have 2 ASINs:
different_names = reviews[reviews["asins"] ==
                          "B00L9EPT8O,B01E6AO69U"]["name"].unique()
for name in different_names:
    print(name)


# In[26]:


##The output confirmed that each ASIN can have multiple names.
##Therefore we should only really concern ourselves with which ASINs do well, not the product names.
fig = plt.figure(figsize=(16,10))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
reviews["asins"].value_counts().plot(kind="bar", ax=ax1, title="ASIN Frequency")
np.log10(reviews["asins"].value_counts()).plot(kind="bar", ax=ax2,
                                               title="ASIN Frequency (Log10 Adjusted)")
plt.show()


# In[27]:



print(reviews["reviews.rating"].mean())

asins_count_ix = reviews["asins"].value_counts().index
plt.subplots(2,1,figsize=(16,12))
plt.subplot(2,1,1)
reviews["asins"].value_counts().plot(kind="bar", title="ASIN Frequency")
plt.subplot(2,1,2)
sns.pointplot(x="asins", y="reviews.rating", order=asins_count_ix, data=reviews)
plt.xticks(rotation=90)
plt.show()


# In[28]:


## Sentiment Analysis
def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"
# Add sentiments to the data
strat_train["Sentiment"] = strat_train["reviews.rating"].apply(sentiments)
strat_test["Sentiment"] = strat_test["reviews.rating"].apply(sentiments)
print(strat_train["Sentiment"][:20])


# In[29]:


print(strat_train["Sentiment"])


# In[30]:


strat_train["Sentiment"].value_counts()


# In[47]:


positive=25823
Neutral=1205
Negative=645
Rating = ['Positive','Negative','Neutral']
 
da = [25823,1205,645]
 
# Creating plot
fig = plt.figure(figsize =(10, 7))
plt.pie(da, labels = Rating)
plt.legend(title = "training data review")
# show plot
plt.show()


# In[48]:


print(strat_test["Sentiment"])


# In[49]:


strat_test["Sentiment"].value_counts()


# In[50]:


positive=25823
Neutral=1205
Negative=645
Rating = ['Positive','Negative','Neutral']
 
da = [25823,1205,645]
 
# Creating plot
fig = plt.figure(figsize =(10, 7))
plt.pie(da, labels = Rating)
plt.legend(title = "testing data review")
# show plot
plt.show()


# In[ ]:




