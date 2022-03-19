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


#Dataframe(DF) oluşturma
veriler=pd.read_csv('maaslar.csv')
print(veriler)

#Bağımlı ve bağımsız değişkenlerin oluşturulması
x=veriler.iloc[:,1:2].values
y=veriler.iloc[:,-1:].values


#Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

'''
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x))
'''

#Polynomial Regression
poly_reg=PolynomialFeatures(degree=7)
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x, y, color="blue")
plt.plot(x, lin_reg2.predict(x_poly),color="red")



























