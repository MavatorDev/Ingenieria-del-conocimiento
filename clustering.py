#!/usr/bin/env python
# coding: utf-8

# In[2]:


from os import listdir


# In[13]:


listdir(".")


# In[5]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[8]:


data=pd.read_csv('energydata.csv')
df=pd.DataFrame(data)
#print(df)
#print(df.keys())


# In[77]:


x=df['T_out']
y=df['RH_out']

X=np.array(list(zip(x,y)))
#print(X)


# In[78]:


kmeans=KMeans(n_clusters=3) 
kmeans=kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_


# In[79]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[80]:


idx = np.random.random() * len(X)
idx


# In[81]:


print(centroids)
colores=["b.","y.","r.","c.","m."]
fig = plt.figure()
for i in range(0, 1000):
    #print("Coordenada: ",X[i]," Label: ",labels[i])
    idx = int(np.random.rand() * len(X))
    plt.plot(X[idx][0],X[idx][1], colores[labels[idx]], markersize=10)

plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=250, linewidths=10,zorder=10, color=("red", "blue", "white"))

fig.savefig('grafica2.png')
plt.show()

