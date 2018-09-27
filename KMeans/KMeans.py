"""
@author: Ridvan

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('customers.csv')

X = data.iloc[:,3:].values


from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
cons = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    cons.append(kmeans.inertia_)

plt.plot(range(1,11),cons)
plt.show()

kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_predict= kmeans.fit_predict(X)
print(Y_predict)  
plt.scatter(X[Y_predict==0,0],X[Y_predict==0,1],s=100, c='red')
plt.scatter(X[Y_predict==1,0],X[Y_predict==1,1],s=100, c='blue')
plt.scatter(X[Y_predict==2,0],X[Y_predict==2,1],s=100, c='green')
plt.scatter(X[Y_predict==3,0],X[Y_predict==3,1],s=100, c='yellow')
plt.title('KMeans')
plt.show()



