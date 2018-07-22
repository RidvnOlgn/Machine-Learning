
"""
@author: Ridvan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("salary.csv")

x = data.iloc[:,1:2].values
y = data.iloc[:,-1:].values

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(x,y)

plt.scatter(x,y,color="red")
plt.plot(x,r_dt.predict(x),color="blue")