import pandas as pd
import numpy as np 
datos=pd.read_csv('energydata.csv')

X= datos[['T8','T_out','Windspeed']]
y= datos[['RH_8','RH_out']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, alpha=0.0001,
                     solver='adam', random_state=21,tol=0.000000001)

mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test.predictions))
