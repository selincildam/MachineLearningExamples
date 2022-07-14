# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:59:21 2022

@author: Selin Çıldam
"""
#import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#Read data
data=pd.read_csv("Wine.csv")

#Get the X and y
X=data.iloc[:,0:13].values
y=data.iloc[:,13].values

#train, test split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)

#Scale the data
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Train the model without PCA
lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_preds=lr.predict(X_test)

#PCA
pca=PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

#Train the model with PCA
lr=LogisticRegression()
lr.fit(X_train_pca,y_train)
y_preds_pca=lr.predict(X_test_pca)

#Confusion Matrixes

print("Model's Confusion Matrix Without PCA")
cm=confusion_matrix(y_test,y_preds)
print(cm)


print("Model's Confusion Matrix With PCA")
cm_pca=confusion_matrix(y_test,y_preds_pca)
print(cm_pca)


print("Compare Models with and without PCA")
cm_pca_=confusion_matrix(y_preds,y_preds_pca)
print(cm_pca_)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda=LDA()
X_train_lda=lda.fit_transform(X_train, y_train)
X_test_lda=lda.transform(X_test)

#Train the model with LDA

lr.fit(X_train_lda,y_train)
y_preds_lda=lr.predict(X_test_lda)

print("Model's Confusion Matrix With LDA")
cm_lda_=confusion_matrix(y_preds,y_preds_lda)
print(cm_lda_)








































