# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 22:40:20 2022

@author: Selin Çıldam
"""

#kütüpanelerin içe aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler



#Dataframe(DF) oluşturma
veriler=pd.read_csv('maaslar.csv')
print(veriler)

#Bağımlı ve bağımsız değişkenlerin oluşturulması
x=veriler.iloc[:,1:2].values
y=veriler.iloc[:,-1:].values


#verilerin ölçeklenmesi
scx=StandardScaler()
x_scaled=scx.fit_transform(x)

scy=StandardScaler()
y_scaled=np.ravel(scy.fit_transform(y.reshape(-1,1)))

#Model oluşturma
svr_reg=SVR(kernel='poly', degree=7, gamma='scale', coef0=0.8) #kernel varsayılan olarak=rbf
svr_reg.fit(x_scaled,y_scaled)

#Modelin görselleştirilmesi
plt.scatter(x_scaled, y_scaled, color="red")
plt.plot(x_scaled,svr_reg.predict(x_scaled),color="blue")
plt.show()



























