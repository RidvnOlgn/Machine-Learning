# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("satislar.csv")

aylar = veriler[["Aylar"]]
satislar = veriler[["Satislar"]]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size = 0.33,random_state = 0)
"""
from sklearn.preprocessing import    StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train,y_train)
    
tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")









