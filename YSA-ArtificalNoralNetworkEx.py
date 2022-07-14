# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 14:12:07 2022

@author: Selin Çıldam
"""

#%%
#Import the library
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#%%
#Read data
data=pd.read_csv("Churn_Modelling.csv")

#%%
#Get the X and y
X=data.iloc[:,3:13].values
y=data.iloc[:,13].values

#%%
#Encode the country column
le=LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])

#%%
#Encode the gender column
X[:,2]=le.fit_transform(X[:,2])

#%%
#One hot encode the geography column
ohe=ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],
                      remainder="passthrough")
X=ohe.fit_transform(X)
X=X[:,1:]

#%%
#Train test split
x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.33, random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#%%
#Create the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(6,kernel_initializer="random_uniform", activation="relu",input_dim=11))
classifier.add(Dense(6,kernel_initializer="random_uniform", activation="relu"))
classifier.add(Dense(1,kernel_initializer="random_uniform", activation="sigmoid"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#%%
classifier.fit(X_train,y_train,epochs=75)
y_pred=classifier.predict(X_test)

#%%
y_pred =(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
print(cm)











