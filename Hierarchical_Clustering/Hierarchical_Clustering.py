
"""
@author: Ridvan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("customers.csv")
x = data.iloc[:,3:].values 

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters = 4,affinity="euclidean",linkage = "ward")

y_pred = ac.fit_predict(x)
print(y_pred)

plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100,c="red")
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100,c="yellow")
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100,c="blue")
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],s=100,c="black")

plt.title('HC')
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()
