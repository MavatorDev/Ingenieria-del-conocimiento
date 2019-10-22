
# graficos embebidos
%matplotlib inline
# importando pandas, numpy y matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importando los datasets de sklearn
from sklearn import datasets

datos = datasets.load_datos()
columnas_deseadas = ["T8", "RH_8"]
datos = full_data[columnas_deseadas]
datos_df['TARGET'] = datos.target

# importando el modelo de regresión lineal
from sklearn.linear_model import LinearRegression

rl = LinearRegression() # Creando el modelo.
rl.fit(datos.data, datos.target) # ajustando el modelo

# haciendo las predicciones
predicciones = rl.predict(datos.data)
predicciones_df = pd.DataFrame(predicciones, columns=['Pred'])
predicciones_df.head() # predicciones de las primeras 5 lineas

# Calculando el desvio
np.mean(datos.target - predicciones)

