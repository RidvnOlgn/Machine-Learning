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

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)
