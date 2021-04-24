#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


df = pd.read_csv("blackjack_main.csv")


# In[3]:


cols = ['blkjck','winloss','plybustbeat','dlbustbeat'] 
for ii in cols:
    df[ii] = df[ii].str.lower()


# In[4]:


n = len(set(df['PlayerId'])) + 1


# In[5]:


num_matches = []
for i in range(n):
    aa = df.loc[df["PlayerId"] == i, "PlayerId"].count()
    num_matches.append(aa)
df['matches_played'] = 0
for i in range(len(df)):
    for j in range(n):
        if df["PlayerId"][i] == j:
            df["matches_played"][i] = num_matches[j]


# In[6]:


def percent(df, feats):
    X = df[feats]
    if X[feats[1]].dtype == object:
        y = X[X[feats[1]]=="win"]
    else:
        y = X[X[feats[1]]==1]
    
    win_per = []
    for i in range(n):
        aa = y.loc[y["PlayerId"] == i, "PlayerId"].count()
        win_per.append(aa)
    df['num_wins'] = 0
    for i in range(len(df)):
        for j in range(n):
            if df["PlayerId"][i] == j:
                df["num_wins"][i] = win_per[j]
    win_per = []
    win_per = df['num_wins']/df['matches_played']
    return win_per


# In[7]:


def strat(d,df):
    df['strat'] = 0
    for j in range(len(df)):
        if d[j] == True:
            df['strat'][j] = 1
    feats = ["PlayerId","strat"]
    strat_per = percent(df,feats)
    return strat_per


# In[8]:


feats = ["PlayerId","winloss"]
df['win_percent'] = percent(df,feats)


# In[9]:


feats = ["PlayerId","blkjck"]
df['bljk_per'] = percent(df,feats)


# In[10]:


a = (df['card3']+df['card4']+df['card5']) < 1 
b = df['ply2cardsum'] > 16
c = df['dealcard1']>0
d = a & b & c
df['strat1_per'] = strat(d,df)


# In[11]:


a = (df['card3']+df['card4']+df['card5']) > 0
b = df['dealcard1'] > 0
c = df['ply2cardsum'] < 12
d = a & b & c
df['strat2_per'] = strat(d,df)


# In[12]:


df['wstrat_per'] = df['strat1_per'] + df['strat2_per']


# In[13]:


df.head()


# In[14]:


df = df.drop(['strat','strat1_per','strat2_per','num_wins'], axis = 1)


# In[15]:


feats = ['win_percent','wstrat_per']
X = df[feats]


# In[16]:


from sklearn.cluster import KMeans
k = 4
kmeans = KMeans(n_clusters = k, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
df['cluster'] = y_kmeans


# In[17]:


plt.scatter(df['win_percent'],df['wstrat_per'],s = 8,c = df['cluster'])


# In[37]:


df1 = pd.read_csv('blackjack.csv')
cols = ['blkjck','winloss','plybustbeat','dlbustbeat'] 
for ii in cols:
    df1[ii] = df1[ii].str.lower()


# In[38]:


n = len(set(df1['PlayerId'])) + 1
num_matches = []
for i in range(n):
    aa = df1.loc[df1["PlayerId"] == i, "PlayerId"].count()
    num_matches.append(aa)
df1['matches_played'] = 0
for i in range(len(df1)):
    for j in range(n):
        if df1["PlayerId"][i] == j:
            df1["matches_played"][i] = num_matches[j]


# In[39]:


feats = ["PlayerId","winloss"]
df1['win_percent'] = percent(df1,feats)


# In[41]:


df1.head()


# In[42]:


feats = ["PlayerId","blkjck"]
df1['bljk_per'] = percent(df1,feats)


# In[43]:


df1.head()


# In[45]:


a = (df1['card3']+df1['card4']+df1['card5']) < 1 
b = df1['ply2cardsum'] > 16
c = df1['dealcard1']>0
d = a & b & c
df1['strat1_per'] = strat(d,df1)


# In[46]:


a = (df1['card3']+df1['card4']+df1['card5']) > 0
b = df1['dealcard1'] > 0
c = df1['ply2cardsum'] < 12
d = a & b & c
df1['strat2_per'] = strat(d,df1)


# In[47]:


df1['wstrat_per'] = df1['strat1_per'] + df1['strat2_per']
df1 = df1.drop(['strat','strat1_per','strat2_per','num_wins'], axis = 1)


# In[48]:


df1.head()


# In[49]:


feats = ['win_percent','wstrat_per']
X_test = df1[feats]


# In[50]:


y_kmeans1 = kmeans.predict(X_test)
df1['cluster'] = y_kmeans1
plt.scatter(df1['win_percent'],df1['wstrat_per'],s = 8,c = df1['cluster'])


# In[22]:


#df.to_csv("out.csv",encoding="UTF-8")


# In[97]:


df['cluster'] = df['cluster'].replace([0,1,2,3],[2,1,0,3])
df1['cluster'] = df1['cluster'].replace([0,1,2,3],[2,1,0,3])


# In[98]:


df['rating'] = (0.45*df['wstrat_per'] + 0.55*df['win_percent'])*(df['cluster'] + 1)


# In[99]:


df['normal_df'] = (df['rating'] - df['rating'].min())/(df['rating'].max() - df['rating'].min())


# In[100]:


df1['rating'] = (0.45*df1['wstrat_per'] + 0.55*df1['win_percent'])*(df1['cluster'] + 1)


# In[101]:


df1['normal_df'] = (df1['rating'] - df1['rating'].min())/(df1['rating'].max() - df1['rating'].min())


# In[106]:


#df.to_csv("bljk_rating_train.csv", encoding = 'UTF-8')
#df1.to_csv("bljk_rating_test.csv", encoding = 'UTF-8')


# In[ ]:




