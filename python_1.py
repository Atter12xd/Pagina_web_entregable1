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

# Convertir fechas en el DataFrame de ventas
ventas_df = convertir_fechas(ventas_df, 'fecha')

# Mostrar información después de la limpieza y conversión de fechas
mostrar_informacion_basica(ventas_df, 'Ventas después de limpieza y conversión de fechas')
mostrar_informacion_basica(usuarios_df, 'Usuarios después de limpieza')

# Cargar datos usando NumPy (opcional)
ventas_numpy = cargar_datos_numpy('ventas.csv')
usuarios_numpy = cargar_datos_numpy('usuarios.csv')

# **Preparación de datos para el sistema de recomendación**
# Ejemplo: Codificar categorías si es necesario para modelos de ML
le = LabelEncoder()
ventas_df['producto_id'] = le.fit_transform(ventas_df['producto_id'])

# Crear características (features) y etiquetas (labels) para un modelo de ejemplo
# Suponiendo que queremos predecir la compra de un producto basándonos en el historial de ventas
X = ventas_df[['producto_id', 'cantidad', 'precio']]  # Características simples
y = ventas_df['comprado']  # Etiqueta: si el producto fue comprado (supuesto)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo simple de clasificación (puedes usar otros modelos más adecuados)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar el modelo (ejemplo básico)
accuracy = modelo.score(X_test, y_test)
print(f"\nExactitud del modelo de recomendación: {accuracy:.2f}")