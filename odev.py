# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:39:40 2022

@author: Selin Çıldam
"""


#kütüpanelerin içe aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder #Etiketleme için
from sklearn.model_selection import train_test_split #train ve test datalarının bölümlenmesi için
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#DF içerisindeki sıcaklık alanını tahmin etmeye çalışma

#Dataframe(DF) oluşturma
veriler=pd.read_csv('odev_tenis.csv')
print(veriler)


#Encoding işlemi nominal verileri sayısal verilere dönüştürmek için kullanılır.

'''
#Label encoder nesnesi oluşturma
le=preprocessing.LabelEncoder()

#veriler içerisinden windy kolonunun alınması
windy=veriler.iloc[:,3:4].values
windy=windy.astype(int)

#windy kolonunun içeriklerine göre sayısal değere dönüştürülmesi
windy[:,0]=le.fit_transform(veriler.iloc[:,3])
print(windy)

#veriler içerisinden play kolonunun alınması
play=veriler.iloc[:,4:].values

#play kolonunun içeriklerine göre sayısal değere dönüştürülmesi
play[:,0]=le.fit_transform(veriler.iloc[:,4])
print(play)
'''

veriler2=veriler.apply(LabelEncoder().fit_transform)

#veriler içerisinden outlook kolonunun alınması
outlook=veriler.iloc[:,0:1].values

#outlook kolonunun one hot encoding işlemi ile sayısal alana dönüştürülmesi
ohe=OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)



#outlook alanının etiketlere göre ayrılmış halde DF'ye dönüştürülmesi
outlook_encoded=pd.DataFrame(data=outlook,index=range(14),columns=['overcast','rainy','sunny'])
print(outlook_encoded)

dfVeriler=pd.concat([outlook_encoded,veriler['humidity'],veriler2['windy'],veriler2['play']],axis=1)

'''
#windy sütununun DF'ye dönüştürülmesi
windy_encoded=pd.DataFrame(data=windy,index=range(14),columns=['windy'])
print(windy_encoded)

#play sütununun DF'ye dönüştürülmesi
play_encoded=pd.DataFrame(data=play, index=range(14),columns=['play'])
print(play_encoded)

humidity=veriler['humidity']

#tüm DFlerin birleştirilmesi
dfVeriler=pd.concat([outlook_encoded,humidity,windy_encoded,play_encoded],axis=1)
print(dfVeriler)

'''



#Tempreture sütununun alınması
tmpr=veriler['temperature']

#train test bölümlemesi, x=dfVeriler isimli DataFrame, y=temprature isimli DataFrame
xtrain,xtest,ytrain,ytest=train_test_split(dfVeriler,tmpr,test_size=0.33,random_state=0)

#Model oluşurma
lr=LinearRegression()
lr.fit(xtrain,ytrain)

tempr_predicted=lr.predict(xtest)


#Backward Elimination
X=np.append(arr=np.ones((14,1)).astype(int), values=dfVeriler,axis=1)
#Xlist=dfVeriler.iloc[:,[0,1,2,3,4,5]].values
model=sm.OLS(endog=tmpr,exog=X).fit()
print(model.summary())



dfVeri=dfVeriler[['overcast','rainy','sunny']]

#train test bölümlemesi, x=dfVeriler isimli DataFrame, y=temprature isimli DataFrame
xtrain,xtest,ytrain,ytest=train_test_split(dfVeri,tmpr,test_size=0.33,random_state=0)

#Model oluşurma
lr=LinearRegression()
lr.fit(xtrain,ytrain)

tempr_predicted=lr.predict(xtest)

X=np.append(arr=np.ones((14,1)).astype(int), values=dfVeri,axis=1)
model=sm.OLS(endog=tmpr,exog=X).fit()
print(model.summary())

