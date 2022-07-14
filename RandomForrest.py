# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 22:40:20 2022

@author: Selin Çıldam
"""

#kütüpanelerin içe aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#Dataframe(DF) oluşturma
veriler=pd.read_csv('maaslar.csv')
print(veriler)

#Bağımlı ve bağımsız değişkenlerin oluşturulması
x=veriler.iloc[:,1:2].values
y=veriler.iloc[:,-1:].values
z=x-0.4
k=x+0.5

#Model oluşturma
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(x,y.ravel())

#Modeli görselleştirme
plt.scatter(x, y)
plt.plot(x,rf_reg.predict(x),color="green")
plt.plot(x,rf_reg.predict(k),color="red")
plt.show()

print(rf_reg.predict([[6.6]]))
print(rf_reg.predict([[11]]))




















