# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:54:58 2022

@author: Selin Çıldam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


#Verileri yükleme
veriler=pd.read_excel('Iris.xls') 
print(veriler)

#bağımlı ve bağımsız değişkenlerin alınması
x=veriler.iloc[:,0:4].values
y=veriler.iloc[:,-1:].values

#train,test bölümlemesi
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=0)

#model oluşturma

# 1. Logistic Regression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)

#tahmin etme
lr_y_pred=lr.predict(xtest)

#Conf. matrix
lr_cm=confusion_matrix(ytest,lr_y_pred)
print("Logistic Regression")
print(lr_cm)

#model oluşturma

# 2. Naive Bayes
gnb=GaussianNB()
gnb.fit(xtrain,ytrain)

#tahmin etme
gnb_y_pred=gnb.predict(xtest)

#Conf. matrix
gnb_cm=confusion_matrix(ytest,gnb_y_pred)
print("Gaussian NB")
print(gnb_cm)

# 3. CategoricalNB 
cnb=GaussianNB()
cnb.fit(xtrain,ytrain)

#tahmin etme
cnb_y_pred=cnb.predict(xtest)

#Conf. matrix
cnb_cm=confusion_matrix(ytest,cnb_y_pred)
print("Categorical NB")
print(cnb_cm)

# 4. K-Nearest Neighbors
knn=KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')
knn.fit(xtrain,ytrain)

#tahmin etme
knn_y_pred=knn.predict(xtest)

#Conf. matrix
knn_cm=confusion_matrix(ytest,knn_y_pred)
print("KNearest Neighbors")
print(knn_cm)

# 5. Support Vector Classifier
svc=SVC(kernel='poly') #kernel fonksiyonu sigmoid seçildiğinde diagondaki değerler değişti.
svc.fit(xtrain,ytrain)

#tahmin etme
svc_y_pred=svc.predict(xtest)

#Conf. matrix
svc_cm=confusion_matrix(ytest,svc_y_pred)
print("Support Vector Classifier")
print(svc_cm)

# 6. Decision Tree
dt=DecisionTreeClassifier(criterion='entropy',splitter='random')
dt.fit(xtrain,ytrain)

#tahmin etme
dt_y_pred=dt.predict(xtest)

#Conf. matrix
dt_cm=confusion_matrix(ytest,dt_y_pred)
print("Decision Tree")
print(dt_cm)

# 7. Random Forest 
rfc=RandomForestClassifier(n_estimators=200,criterion='entropy')
rfc.fit(xtrain,ytrain)

#tahmin etme
rfc_y_pred=rfc.predict(xtest)

#Conf. matrix
rfc_cm=confusion_matrix(ytest,rfc_y_pred)
print("Random Forest")
print(rfc_cm)













