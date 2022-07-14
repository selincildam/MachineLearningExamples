# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:35:19 2022

@author: Selin Çıldam
"""


#kütüpanelerin içe aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#Verileri yükleme
veriler=pd.read_csv('veriler.csv')
print(veriler)

#bağımlı ve bağımsız değişkenlerin alınması
x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,-1:].values

#train,test bölümlemesi
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin ölçeklenmesi
#train verilerinin ölçeklenmesi
sctrain=StandardScaler()
xtrain_scaled=sctrain.fit_transform(xtrain)

#test verilerinin ölçeklenmesi
sctest=StandardScaler()
xtest_scaled=sctest.fit_transform(xtest)

#Model oluşturma
rfc=RandomForestClassifier()
rfc.fit(xtrain,ytrain)

#Tahmin
pred=rfc.predict(xtest)
print(pred)

print("Random Forest with CM")
print(confusion_matrix(ytest,pred))












