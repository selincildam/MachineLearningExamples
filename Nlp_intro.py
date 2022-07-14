# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:26:17 2022

@author: Selin Çıldam
"""

#%%
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#%%

nltk.download('stopwords')



#%%

yorumlar=pd.read_csv("C:\MachineLearningCourse\Restaurant_Reviews.csv", on_bad_lines='skip')
ps=PorterStemmer()

#%%
rev_list=[]

for i in range(len(yorumlar)):
    yorum=re.sub('[^a-zA-a]',' ',yorumlar["Review"][i])
    yorum=yorum.lower()
    yorum=yorum.split()
    yorum=[ps.stem(word) for word in yorum if not word in stopwords.words('english')]
    yorum=" ".join(yorum)
    rev_list.append(yorum)


#%%

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)

X=cv.fit_transform(rev_list).toarray()
y=yorumlar.iloc[:,1].values


























