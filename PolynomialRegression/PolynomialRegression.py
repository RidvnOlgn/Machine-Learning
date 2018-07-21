
"""
@author: Ridvan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

X = x.values
Y = y.values


#for 2. degree

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
plt.scatter(X,Y,color = "red")
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)),color = "blue")
plt.show()

#for 4. degree

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
plt.scatter(X,Y,color = "red")
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)),color = "blue")
plt.show()
