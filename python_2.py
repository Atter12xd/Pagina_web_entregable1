import sys
import io

# Configura la salida estándar y de errores a UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import os
import sys
if sys.platform == "win32":
    os.system('chcp 65001')  # Cambia la codificación a UTF-8

# Paso 2: Importar Librerías Necesarias
# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Paso 3: Funciones para Cargar y Preparar los Datos
def cargar_datos_pandas(archivo_ventas, archivo_usuarios):
    ventas_df = pd.read_csv(archivo_ventas)
    usuarios_df = pd.read_csv(archivo_usuarios)
    return ventas_df, usuarios_df

def limpiar_datos(dataframe):
    dataframe.dropna(inplace=True)
    return dataframe

def convertir_fechas(dataframe, columna_fecha):
    if columna_fecha in dataframe.columns:
        dataframe[columna_fecha] = pd.to_datetime(dataframe[columna_fecha])
    else:
        print(f"La columna '{columna_fecha}' no existe en el DataFrame.")
    return dataframe

#Paso 4: Cargar y Limpiar los Datos
ventas_df, usuarios_df = cargar_datos_pandas('ventas.csv', 'usuarios.csv')
ventas_df = limpiar_datos(ventas_df)
usuarios_df = limpiar_datos(usuarios_df)
ventas_df = convertir_fechas(ventas_df, 'fecha')

#Paso 5: Preparar los Datos para el Modelo
le = LabelEncoder()
ventas_df['producto'] = le.fit_transform(ventas_df['producto'])

X = ventas_df[['producto', 'cantidad', 'monto']]
y = (ventas_df['monto'] > 100).astype(int)  # Etiqueta binaria simple

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Paso 6: Clasificación con K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Exactitud del modelo KNN: {accuracy_knn:.2f}")

# Paso 7: Clustering con K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
ventas_df['cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
plt.scatter(X['producto'], X['monto'], c=ventas_df['cluster'])
plt.xlabel('Producto')
plt.ylabel('Monto')
plt.title('Clusters de Productos')
plt.show()