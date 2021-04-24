#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


pd.read_csv("Poker.csv").head(10)


# In[3]:


n = 0

def proc_file(filename):
    df = pd.read_csv(filename)
    cols = ['Blind','Preflop','Flop','Turn','River','Showdown'] 
    for ii in cols:
        df[ii] = df[ii].str.lower()
    x = df['Flop'].str.startswith('$')
    for i in range(len(x)):
        if x[i] == True:
            df['Flop'][i] = 'won' 
    a = df['Turn'].str.startswith('$')
    for i in range(len(a)):
        if a[i] == True:
            df['Turn'][i] = 'won'         
    b = df['River'].str.startswith('$')
    for i in range(len(b)):
        if b[i] == True:
            df['River'][i] = 'won'         
    c = df['Showdown'].str.startswith('$')
    for i in range(len(c)):
        if c[i] == True:
            df['Showdown'][i] = 'won' 
    
    f = (df['Preflop'].str.startswith('ra') +
         df['Preflop'].str.startswith('f'))  
    df['PFRF'] = 0
    for j in range(len(df)):
        if f[j] == True:
            df['PFRF'][j] = 1 
    
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    enc_df = pd.DataFrame(enc.fit_transform(df[['Player_ID']]))
    df = df.join(enc_df)
    n = len(set(df[0]))        
    
    num_matches = []
    for i in range(n):
        aa = df.loc[df[0] == i, 0].count()
        num_matches.append(aa)
    df['matches_played'] = 0
    for i in range(len(df)):
        for j in range(n):
            if df[0][i] == j:
                df["matches_played"][i] = num_matches[j]         
    
    pfr_percent = []
    feats = [0,"PFRF"]
    X = df[feats]
    y = X[X["PFRF"]==1]
    for i in range(n):
        aa = y.loc[y[0] == i, 0].count()
        pfr_percent.append(aa)
    df['matches_pfrf'] = 0
    for i in range(len(df)):
        for j in range(n):
            if df[0][i] == j:
                df["matches_pfrf"][i] = pfr_percent[j]     
   
    df['pfrf_percent'] = df['matches_pfrf']/df['matches_played']
    feats = [0,"Winner"]
    X = df[feats]
    y = X[X["Winner"]==True]
    win_per = []
    for i in range(n):
        aa = y.loc[y[0] == i, 0].count()
        win_per.append(aa)
    df['num_wins'] = 0
    for i in range(len(df)):
        for j in range(n):
            if df[0][i] == j:
                df["num_wins"][i] = win_per[j]
    
    df['win_percent'] = df['num_wins']/df['matches_played']
    
    g = (df['Preflop'].str.startswith('ca') + df['Preflop'].str.startswith('b')+ 
         df['Preflop'].str.startswith('r') + df['Preflop'].str.startswith('a'))
    h = ( df['Flop'].str.startswith('ca') + df['Flop'].str.startswith('b') +
         df['Flop'].str.startswith('r') + df['Flop'].str.startswith('a'))
    gg = (df['Turn'].str.startswith('ca') + df['Turn'].str.startswith('b') +
          df['Turn'].str.startswith('r') + df['Turn'].str.startswith('a'))
    gh = (df['River'].str.startswith('ca') + df['River'].str.startswith('b') +
          df['River'].str.startswith('r') + df['River'].str.startswith('a'))
    fe = g+h+gg+gh
    df['c_t_bets'] = 0
    for j in range(len(df)):
        if fe[j] == True:
            df['c_t_bets'][j] = 1
    feats = [0,"c_t_bets"]
    X = df[feats]
    y = X[X["c_t_bets"]==1]   

    VPIP = []
    for i in range(n):
        aa = y.loc[y[0] == i, 0].count()
        VPIP.append(aa) 
    df['c_t_bets_per'] = 0
    for i in range(len(df)):
        for j in range(n):
            if df[0][i] == j:
                df['c_t_bets_per'][i] = VPIP[j]    
    
    df['s_rat'] = df['c_t_bets_per']/df['matches_played']         
    df = df.drop(['c_t_bets'], axis = 1)
    df = df.rename({0:'PlayerID'}, axis = 1)
    
    return df


# In[4]:


def axe_3d_plot(df):
    fig1 = plt.figure(figsize = (10,10))
    ax1 = fig1.add_subplot(111,projection='3d')
    ax1.scatter(df['s_rat'],df['win_percent'],df['pfrf_percent'],s=8,c = df['cluster'])
    ax1.set_xlabel("VPMIP_percent")
    ax1.set_ylabel("Win percent")
    ax1.set_zlabel("pfrf_percent")
    plt.show()    


# In[5]:


df = proc_file("Poker.csv") 


# In[6]:


df


# In[7]:


fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(df['s_rat'],df['win_percent'],df['pfrf_percent'],s=5)
plt.show()


# In[8]:


feats = ['pfrf_percent','win_percent',
         's_rat']
X = df[feats]


# In[9]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,16):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,16),wcss)
plt.title('cluster analysis')
plt.show


# In[10]:


k = 5
plt.figure(figsize =((8,8)))
           
kmeans = KMeans(n_clusters = k, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
df['cluster'] = y_kmeans
axe_3d_plot(df)


# In[12]:


df['cluster'] = df['cluster'].replace([0,1,2,3,4],[0,3,4,2,1])


# In[13]:


df1 = proc_file("Poker_test.csv")
X1 = df1[feats]
y_kmeans1 = kmeans.predict(X1)
df1['cluster'] = y_kmeans1


# In[14]:


df1['cluster'] = df1['cluster'].replace([0,1,2,3,4],[0,3,4,2,1])


# In[15]:


axe_3d_plot(df1)


# In[12]:


#df2 = proc_file("Poker_test_1.csv")
#X2 = df2[feats]
#y_kmeans2 = kmeans.predict(X2)
#df2['cluster'] = y_kmeans2


# In[13]:


#axe_3d_plot(df2)


# In[14]:


#df3 = proc_file("Poker_test_2.csv")
#X3 = df3[feats]
#y_kmeans3 = kmeans.predict(X3)
#df3['cluster'] = y_kmeans3


# In[15]:


#axe_3d_plot(df3)


# In[16]:


#df4 = proc_file("Poker_test_3.csv")
#X4 = df4[feats]
#y_kmeans4 = kmeans.predict(X4)
#df4['cluster'] = y_kmeans4


# In[17]:


#axe_3d_plot(df4)


# In[18]:


#df5 = proc_file("Poker_test_4.csv")
#X5 = df5[feats]
#y_kmeans5 = kmeans.predict(X5)
#df5['cluster'] = y_kmeans5


# In[19]:


#axe_3d_plot(df5)


# In[23]:


import pickle
filename = 'finalized_model.pkl'
pickle.dump(kmeans, open(filename, 'wb'))


# In[24]:


#model = pickle.load(open(filename, 'rb'))


# In[16]:


df['rating'] =  (0.15*df['pfrf_percent']+ 0.25*df['win_percent'] +0.15*df['s_rat'])*(df['cluster']+1)
df['normal_rating'] = (df['rating'] - df['rating'].min())/(df['rating'].max() - df['rating'].min())


# In[17]:


df1['rating'] =  (0.15*df1['pfrf_percent']+ 0.25*df1['win_percent'] +0.15*df1['s_rat'])*(df1['cluster']+1)
df1['normal_rating'] = (df1['rating'] - df1['rating'].min())/(df1['rating'].max() - df1['rating'].min())


# In[22]:


set(df1['normal_rating'][df1['cluster']==4])


# In[23]:


#df.to_csv("MM1.csv", encoding = "UTF-8", index=False)
#df1.to_csv("instance1.csv", encoding = "UTF-8", index=False)


# In[ ]:




