
"""
@author: Ridvan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("veri.csv")

x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

#for linear

svc = SVC(kernel="linear")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

#for rbf
svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

#for poly
svc = SVC(kernel="poly")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)


