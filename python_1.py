# Empezando nuevo proyecto para el sistema de recomendación de productos
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # Ejemplo de modelo para recomendaciones
import matplotlib.pyplot as plt
import seaborn as sns

# Función para cargar los datos desde archivos CSV usando Pandas
def cargar_datos_pandas(archivo_ventas, archivo_usuarios):
    ventas_df = pd.read_csv(archivo_ventas)
    usuarios_df = pd.read_csv(archivo_usuarios)
    return ventas_df, usuarios_df

# Función para cargar los datos desde archivos CSV usando NumPy (opcional para otros análisis)
def cargar_datos_numpy(archivo):
    return np.genfromtxt(archivo, delimiter=',', skip_header=1)

# Función para limpiar los datos eliminando filas con valores faltantes
def limpiar_datos(dataframe):
    dataframe.dropna(inplace=True)
    return dataframe

# Función para convertir fechas en un DataFrame si la columna 'fecha' existe
def convertir_fechas(dataframe, columna_fecha):
    if columna_fecha in dataframe.columns:
        dataframe[columna_fecha] = pd.to_datetime(dataframe[columna_fecha])
    else:
        print(f"La columna {columna_fecha} no existe en el DataFrame.")
    return dataframe

# Función para mostrar las primeras filas y los tipos de datos de un DataFrame
def mostrar_informacion_basica(dataframe, nombre):
    print(f"\nDatos de {nombre}:")
    print(dataframe.head())
    print(f"\nTipos de Datos en {nombre}:")
    print(dataframe.dtypes)
    
# Cargar datos usando Pandas
ventas_df, usuarios_df = cargar_datos_pandas('ventas.csv', 'usuarios.csv')

# Mostrar información básica de los datos cargados
mostrar_informacion_basica(ventas_df, 'Ventas')
mostrar_informacion_basica(usuarios_df, 'Usuarios')

# Limpiar los datos
ventas_df = limpiar_datos(ventas_df)
usuarios_df = limpiar_datos(usuarios_df)