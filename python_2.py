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
