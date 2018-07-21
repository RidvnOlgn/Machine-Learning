

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



veriler = pd.read_csv('data.csv')

Yas = veriler.iloc[:,1:4].values



#encoder:  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()


c = veriler.iloc[:,-1:].values
c[:,0] = le.fit_transform(c[:,0])
c=ohe.fit_transform(c).toarray()



sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])

cinsiyet = veriler.iloc[:,-1].values

sonuc3 = pd.DataFrame(data = c[:,:1] , index=range(22), columns=['cinsiyet'])


s=pd.concat([sonuc,sonuc2],axis=1)

s2= pd.concat([s,sonuc3],axis=1)



from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)











