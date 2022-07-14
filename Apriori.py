# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:33:55 2022

@author: Selin Çıldam

"""
#%%
import pandas as pd
import numpy as np

veriler=pd.read_csv('sepet.csv', header=None)


#%%
from apyori import apriori

#%%
t=[]

for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range(20)])
    
#%%
rules = apriori(t, min_support=0.01 , min_confidence=0.2, min_lift=3,min_length=2)

#%%
print(list(rules))
