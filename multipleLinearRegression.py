# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:47:01 2022

@author: Selin Çıldam
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:10:02 2022

@author: Selin Çıldam
"""

#kütüpanelerin içe aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #train ve test datalarının bölümlenmesi için
from sklearn import preprocessing #Etiketleme için
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#Verileri yükleme
veriler=pd.read_csv('veriler.csv')
print(veriler)

#boy-kilo-yas alanlarının alınması
boyKiloYas=veriler.iloc[:,1:4].values
print(boyKiloYas)

#veriler içerisinden ülke kolonunun alınması
ulke=veriler.iloc[:,0:1].values

#ülke adlarının içeriklerine göre sayısal değere dönüştürülmesi
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

#Her bir değer için sütun oluşturulması
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)


#veriler içerisinden cinsiyet kolonunun alınması
c=veriler.iloc[:,-1:].values

#cinsiyet alanının sayısal değere dönüştürülmesi
le=preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)

#Her bir değer için sütun oluşturulması
ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

#ülke kodlarının etiketlere göre ayrılmış halde DF'ye dönüştürülmesi
s=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(s)

#boy-kilo-yas sütunlarının DF'ye dönüştürülmesi
s2=pd.DataFrame(data=boyKiloYas,index=range(22),columns=['boy','kilo','yas'])
print(s2)

#Cinsiyet sütununun DF'ye dönüştürülmesi

s3=pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])

#DF'lerin birleştirilmesi
ss=pd.concat([s,s2],axis=1)
sson=pd.concat([ss,s3],axis=1)

print(sson)

#train test bölümlemesi, x=ss isimli DataFrame, y=s3 isimli DataFrame
xtrain,xtest,ytrain,ytest=train_test_split(ss,s3,test_size=0.33,random_state=0)

#model oluşturma
lr=LinearRegression()
lr.fit(xtrain,ytrain)

#tahmin
y_pred=lr.predict(xtest)

print(lr.score(xtest,ytest))

#Veriler DF üzerinden boy alanını alıp, verilerdeki boy sütununu düşürme
boy=sson.iloc[:,3:4].values
veri=sson.drop(['boy'],axis=1)

#train test bölümlemesi,
Xtrain,Xtest,Ytrain,Ytest=train_test_split(veri,boy,test_size=0.33,random_state=0)

lr.fit(Xtrain,Ytrain)

#tahmin
y_pred=lr.predict(Xtest)

print(lr.score(Xtest,Ytest))

#Statmodels ile backward elimination

#Veri dizisi başında 1lerden oluşan bir sabit ekleme
X=np.append(arr=np.ones((22,1)).astype(int), values=veri,axis=1)
Xlist=veri.iloc[:,[0,1,2,3,4,5]].values

#Xlistin np.array e çevrilmesi
Xlist=np.array(Xlist,dtype=float)

model=sm.OLS(boy,Xlist).fit()
print(model.summary())


'''
#verileri sıralama
xtrain=xtrain.sort_index()
ytrain=ytrain.sort_index()

#verileri görselleştirme
plt.plot(xtrain,ytrain)
plt.plot(xtest, tahmin)
plt.title("Aylara Göre Satış Tahmini")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")


'''
