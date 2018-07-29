# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 20:19:58 2018

@author: Ridvan
"""

import numpy as np
import pandas as pd

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

#decisiontree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="gini")
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_pred,y_test)
print(cm)

dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_pred,y_test)
print(cm)


#randomforest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10,criterion="gini")
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

cm = confusion_matrix(y_pred,y_test)
print(cm)


rfc = RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

cm = confusion_matrix(y_pred,y_test)
print(cm)









