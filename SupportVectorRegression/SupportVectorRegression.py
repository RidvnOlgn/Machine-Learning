
"""
@author: Ridvan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("salary.csv")

x = data.iloc[:,1:2].values
y = data.iloc[:,-1:].values

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_sc = sc1.fit_transform(x)

sc2 = StandardScaler()
y_sc = sc2.fit_transform(y)


from sklearn.svm import SVR

#for rbf

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_sc,y_sc)

plt.scatter(x_sc,y_sc,color = "red")
plt.plot(x_sc,svr_reg.predict(x_sc),color="blue")

#for linear
svr_reg = SVR(kernel = "linear")
svr_reg.fit(x_sc,y_sc)

plt.plot(x_sc,svr_reg.predict(x_sc),color="green")


#for polynomial
svr_reg = SVR(kernel = "poly")
svr_reg.fit(x_sc,y_sc)

plt.plot(x_sc,svr_reg.predict(x_sc),color="black")