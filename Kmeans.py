# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:02:52 2022

@author: Selin Çıldam
"""

#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read the data
veriler=pd.read_csv('musteriler.csv')


#Create new array
#%%
x=veriler.iloc[:,3:].values

#İmport the KMeans library
#%%
from sklearn.cluster import KMeans

#Create a model
#%%
kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(x)

print(kmeans.cluster_centers_)

#%%
sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++', random_state=123)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)
 
print(sonuclar)

#%%
plt.plot(range(1,11),sonuclar)
plt.show()
















