"""
Ejercicio 1: Análisis del Dataset Titanic
Objetivo: Preparar los datos para un modelo que prediga la supervivencia de los pasajeros.

Etapas:
1. Carga del dataset
2. Eliminación de columnas irrelevantes
3. Manejo de valores nulos
4. Codificación de variables categóricas
5. Estandarización de variables numéricas
6. División en conjuntos de entrenamiento y prueba
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class TitanicProcessor:
    """Clase para procesar el dataset Titanic"""
    
    def __init__(self):
        self.df_original = None
        self.df_procesado = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.columnas_eliminadas = []
        self.info_nulos = {}
        
    def cargar_datos(self):
        """
        1. Carga del dataset con pandas
        """
        ruta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_csv = os.path.join(ruta_actual, 'titanic.csv')
        
        self.df_original = pd.read_csv(ruta_csv)
        self.df_procesado = self.df_original.copy()
        
        return {
            'filas': self.df_original.shape[0],
            'columnas': self.df_original.shape[1],
            'columnas_nombres': list(self.df_original.columns),
            'tipos_datos': self.df_original.dtypes.to_dict()
        }
    
    def explorar_datos(self):
        """
        2. Exploración inicial del dataset
        """
        info = {
            'shape': self.df_original.shape,
            'columnas': list(self.df_original.columns),
            'tipos': self.df_original.dtypes.to_dict(),
            'nulos': self.df_original.isnull().sum().to_dict(),
            'nulos_porcentaje': (self.df_original.isnull().sum() / len(self.df_original) * 100).to_dict(),
            'duplicados': self.df_original.duplicated().sum(),
            'estadisticas': self.df_original.describe().to_dict()
        }
        
        return info
    
    def eliminar_columnas_irrelevantes(self):
        """
        2. Elimine columnas irrelevantes como Name, Ticket o Cabin
        """
        columnas_a_eliminar = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        self.columnas_eliminadas = [col for col in columnas_a_eliminar if col in self.df_procesado.columns]
        
        self.df_procesado = self.df_procesado.drop(columns=self.columnas_eliminadas, errors='ignore')
        
        return {
            'eliminadas': self.columnas_eliminadas,
            'columnas_restantes': list(self.df_procesado.columns),
            'shape_nuevo': self.df_procesado.shape
        }
    
    def manejar_valores_nulos(self):
        """
        3. Verifique valores nulos y reemplácelos con la media o moda según corresponda
        """
        self.info_nulos = {
            'antes': self.df_procesado.isnull().sum().to_dict()
        }
        
        # Age: reemplazar con la mediana (variable numérica)
        if 'Age' in self.df_procesado.columns:
            media_age = self.df_procesado['Age'].median()
            self.df_procesado['Age'] = self.df_procesado['Age'].fillna(media_age)
            self.info_nulos['Age_reemplazo'] = f'Mediana: {media_age:.2f}'
        
        # Fare: reemplazar con la mediana
        if 'Fare' in self.df_procesado.columns:
            media_fare = self.df_procesado['Fare'].median()
            self.df_procesado['Fare'] = self.df_procesado['Fare'].fillna(media_fare)
            self.info_nulos['Fare_reemplazo'] = f'Mediana: {media_fare:.2f}'
        
        # Embarked: reemplazar con la moda (variable categórica)
        if 'Embarked' in self.df_procesado.columns:
            moda_embarked = self.df_procesado['Embarked'].mode()[0]
            self.df_procesado['Embarked'] = self.df_procesado['Embarked'].fillna(moda_embarked)
            self.info_nulos['Embarked_reemplazo'] = f'Moda: {moda_embarked}'
        
        # Eliminar cualquier otra fila con valores nulos restantes
        self.df_procesado = self.df_procesado.dropna()
        
        self.info_nulos['despues'] = self.df_procesado.isnull().sum().to_dict()
        self.info_nulos['filas_eliminadas'] = self.df_original.shape[0] - self.df_procesado.shape[0]
        
        return self.info_nulos
    
    def codificar_variables(self):
        """
        4. Codifique las variables Sex y Embarked
        """
        info_codificacion = {}
        
        # Codificar Sex (male=1, female=0)
        if 'Sex' in self.df_procesado.columns:
            le_sex = LabelEncoder()
            valores_originales_sex = self.df_procesado['Sex'].unique().tolist()
            self.df_procesado['Sex'] = le_sex.fit_transform(self.df_procesado['Sex'])
            self.label_encoders['Sex'] = le_sex
            info_codificacion['Sex'] = {
                'valores_originales': valores_originales_sex,
                'valores_codificados': self.df_procesado['Sex'].unique().tolist(),
                'mapeo': dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))
            }
        
        # Codificar Embarked (S, C, Q)
        if 'Embarked' in self.df_procesado.columns:
            le_embarked = LabelEncoder()
            valores_originales_embarked = self.df_procesado['Embarked'].unique().tolist()
            self.df_procesado['Embarked'] = le_embarked.fit_transform(self.df_procesado['Embarked'])
            self.label_encoders['Embarked'] = le_embarked
            info_codificacion['Embarked'] = {
                'valores_originales': valores_originales_embarked,
                'valores_codificados': self.df_procesado['Embarked'].unique().tolist(),
                'mapeo': dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))
            }
        
        return info_codificacion
    
    def estandarizar_variables(self):
        """
        5. Estandarice las variables numéricas (Age, Fare)
        """
        columnas_a_estandarizar = ['Age', 'Fare']
        columnas_disponibles = [col for col in columnas_a_estandarizar if col in self.df_procesado.columns]
        
        info_estandarizacion = {
            'columnas': columnas_disponibles,
            'estadisticas_antes': {},
            'estadisticas_despues': {}
        }
        
        # Guardar estadísticas antes de estandarizar
        for col in columnas_disponibles:
            info_estandarizacion['estadisticas_antes'][col] = {
                'media': float(self.df_procesado[col].mean()),
                'std': float(self.df_procesado[col].std()),
                'min': float(self.df_procesado[col].min()),
                'max': float(self.df_procesado[col].max())
            }
        
        # Estandarizar
        if columnas_disponibles:
            self.df_procesado[columnas_disponibles] = self.scaler.fit_transform(
                self.df_procesado[columnas_disponibles]
            )
        
        # Guardar estadísticas después de estandarizar
        for col in columnas_disponibles:
            info_estandarizacion['estadisticas_despues'][col] = {
                'media': float(self.df_procesado[col].mean()),
                'std': float(self.df_procesado[col].std()),
                'min': float(self.df_procesado[col].min()),
                'max': float(self.df_procesado[col].max())
            }
        
        return info_estandarizacion
    
    def dividir_datos(self, test_size=0.3, random_state=42):
        """
        6. Divida los datos en entrenamiento (70%) y prueba (30%)
        """
        # Separar características (X) y variable objetivo (y)
        X = self.df_procesado.drop('Survived', axis=1)
        y = self.df_procesado['Survived']
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        info_division = {
            'total_registros': len(self.df_procesado),
            'caracteristicas': list(X.columns),
            'variable_objetivo': 'Survived',
            'train_shape': {
                'X_train': self.X_train.shape,
                'y_train': self.y_train.shape
            },
            'test_shape': {
                'X_test': self.X_test.shape,
                'y_test': self.y_test.shape
            },
            'porcentajes': {
                'train': f'{(1-test_size)*100:.0f}%',
                'test': f'{test_size*100:.0f}%'
            },
            'distribucion_clases_train': self.y_train.value_counts().to_dict(),
            'distribucion_clases_test': self.y_test.value_counts().to_dict()
        }
        
        return info_division
    
    def obtener_primeros_registros(self, n=5):
        """
        Obtener los primeros N registros procesados
        """
        return self.df_procesado.head(n)
    
    def obtener_dataframe_original(self, n=5):
        """
        Obtener los primeros N registros del dataset original
        """
        return self.df_original.head(n)
    
    def obtener_resumen_completo(self):
        """
        Obtener un resumen completo de todo el procesamiento
        """
        return {
            'dataset_original': {
                'shape': self.df_original.shape,
                'columnas': list(self.df_original.columns)
            },
            'dataset_procesado': {
                'shape': self.df_procesado.shape,
                'columnas': list(self.df_procesado.columns)
            },
            'columnas_eliminadas': self.columnas_eliminadas,
            'valores_nulos_manejados': self.info_nulos,
            'codificacion_aplicada': list(self.label_encoders.keys()),
            'division_datos': {
                'X_train': self.X_train.shape if self.X_train is not None else None,
                'X_test': self.X_test.shape if self.X_test is not None else None,
                'y_train': self.y_train.shape if self.y_train is not None else None,
                'y_test': self.y_test.shape if self.y_test is not None else None
            }
        }


def procesar_titanic_completo():
    """
    Función principal que ejecuta todo el procesamiento del dataset Titanic
    """
    processor = TitanicProcessor()
    
    # 1. Cargar datos
    info_carga = processor.cargar_datos()
    
    # 2. Explorar datos
    info_exploracion = processor.explorar_datos()
    
    # 3. Eliminar columnas irrelevantes
    info_eliminacion = processor.eliminar_columnas_irrelevantes()
    
    # 4. Manejar valores nulos
    info_nulos = processor.manejar_valores_nulos()
    
    # 5. Codificar variables categóricas
    info_codificacion = processor.codificar_variables()
    
    # 6. Estandarizar variables numéricas
    info_estandarizacion = processor.estandarizar_variables()
    
    # 7. Dividir en train/test
    info_division = processor.dividir_datos(test_size=0.3, random_state=42)
    
    # Resumen completo
    resumen = {
        'carga': info_carga,
        'exploracion': info_exploracion,
        'eliminacion': info_eliminacion,
        'nulos': info_nulos,
        'codificacion': info_codificacion,
        'estandarizacion': info_estandarizacion,
        'division': info_division,
        'df_original': processor.obtener_dataframe_original(5),
        'df_procesado': processor.obtener_primeros_registros(5),
        'X_train': processor.X_train,
        'X_test': processor.X_test,
        'y_train': processor.y_train,
        'y_test': processor.y_test
    }
    
    return processor, resumen
