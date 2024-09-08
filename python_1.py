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
