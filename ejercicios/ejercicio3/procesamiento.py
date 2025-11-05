"""
Ejercicio 3: Dataset Iris
Objetivo: Implementar un flujo completo de preprocesamiento y visualizar resultados.

Etapas:
1. Carga del dataset desde sklearn.datasets
2. Conversión a DataFrame con nombres de columnas
3. Estandarización con StandardScaler()
4. División en entrenamiento (70%) y prueba (30%)
5. Visualización de características por clase
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class IrisProcessor:
    """Clase para procesar el dataset Iris"""
    
    def __init__(self):
        self.iris_data = None
        self.df_original = None
        self.df_procesado = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        
    def cargar_datos(self):
        """
        1. Cargue el dataset desde sklearn.datasets
        """
        self.iris_data = load_iris()
        
        return {
            'n_samples': self.iris_data.data.shape[0],
            'n_features': self.iris_data.data.shape[1],
            'feature_names': self.iris_data.feature_names,
            'target_names': self.iris_data.target_names.tolist(),
            'descripcion': self.iris_data.DESCR
        }
    
    def convertir_a_dataframe(self):
        """
        2. Conviértalo en un DataFrame y agregue los nombres de las columnas
        """
        # Crear DataFrame con las características
        self.df_original = pd.DataFrame(
            data=self.iris_data.data,
            columns=self.iris_data.feature_names
        )
        
        # Agregar la columna target
        self.df_original['target'] = self.iris_data.target
        
        # Agregar columna con el nombre de la especie
        self.df_original['species'] = self.df_original['target'].map(
            {i: name for i, name in enumerate(self.iris_data.target_names)}
        )
        
        self.feature_names = self.iris_data.feature_names
        self.target_names = self.iris_data.target_names
        
        return {
            'shape': self.df_original.shape,
            'columnas': list(self.df_original.columns),
            'feature_names': self.feature_names,
            'target_names': self.target_names.tolist(),
            'primeros_registros': self.df_original.head()
        }
    
    def explorar_datos(self):
        """
        Exploración del dataset
        """
        info = {
            'estadisticas_descriptivas': self.df_original.describe().to_dict(),
            'distribucion_clases': self.df_original['target'].value_counts().to_dict(),
            'distribucion_especies': self.df_original['species'].value_counts().to_dict(),
            'info_clases': {}
        }
        
        # Información por cada clase
        for i, species in enumerate(self.target_names):
            clase_data = self.df_original[self.df_original['target'] == i]
            info['info_clases'][species] = {
                'cantidad': len(clase_data),
                'porcentaje': f"{(len(clase_data) / len(self.df_original) * 100):.2f}%"
            }
        
        return info
    
    def estandarizar_datos(self):
        """
        3. Aplique estandarización con StandardScaler()
        """
        # Crear copia para procesamiento
        self.df_procesado = self.df_original.copy()
        
        # Separar características para estandarizar (sin target y species)
        features_to_scale = [col for col in self.df_procesado.columns 
                            if col not in ['target', 'species']]
        
        info_estandarizacion = {
            'columnas_estandarizadas': features_to_scale,
            'estadisticas_antes': {},
            'estadisticas_despues': {}
        }
        
        # Guardar estadísticas antes de estandarizar
        for col in features_to_scale:
            info_estandarizacion['estadisticas_antes'][col] = {
                'media': float(self.df_procesado[col].mean()),
                'std': float(self.df_procesado[col].std()),
                'min': float(self.df_procesado[col].min()),
                'max': float(self.df_procesado[col].max())
            }
        
        # Aplicar estandarización
        self.df_procesado[features_to_scale] = self.scaler.fit_transform(
            self.df_procesado[features_to_scale]
        )
        
        # Guardar estadísticas después de estandarizar
        for col in features_to_scale:
            info_estandarizacion['estadisticas_despues'][col] = {
                'media': float(self.df_procesado[col].mean()),
                'std': float(self.df_procesado[col].std()),
                'min': float(self.df_procesado[col].min()),
                'max': float(self.df_procesado[col].max())
            }
        
        return info_estandarizacion
    
    def dividir_datos(self, test_size=0.3, random_state=42):
        """
        4. Divida el dataset (70% entrenamiento, 30% prueba)
        """
        # Separar características (X) y variable objetivo (y)
        feature_cols = [col for col in self.df_procesado.columns 
                       if col not in ['target', 'species']]
        
        X = self.df_procesado[feature_cols]
        y = self.df_procesado['target']
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        info_division = {
            'total_registros': len(self.df_procesado),
            'caracteristicas': list(X.columns),
            'num_caracteristicas': len(X.columns),
            'variable_objetivo': 'target',
            'clases': self.target_names.tolist(),
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
    
    def obtener_datos_visualizacion(self):
        """
        Preparar datos para visualización (sepal length vs petal length por clase)
        """
        # Usar nombres simplificados para las columnas
        sepal_length_col = [col for col in self.df_original.columns if 'sepal length' in col][0]
        petal_length_col = [col for col in self.df_original.columns if 'petal length' in col][0]
        
        datos_viz = {
            'original': {
                'sepal_length': self.df_original[sepal_length_col].tolist(),
                'petal_length': self.df_original[petal_length_col].tolist(),
                'target': self.df_original['target'].tolist(),
                'species': self.df_original['species'].tolist()
            },
            'estandarizado': {
                'sepal_length': self.df_procesado[sepal_length_col].tolist(),
                'petal_length': self.df_procesado[petal_length_col].tolist(),
                'target': self.df_procesado['target'].tolist(),
                'species': self.df_procesado['species'].tolist()
            },
            'nombres_columnas': {
                'sepal_length': sepal_length_col,
                'petal_length': petal_length_col
            }
        }
        
        return datos_viz
    
    def obtener_estadisticas_descriptivas(self):
        """
        Obtener estadísticas descriptivas del dataset estandarizado
        """
        feature_cols = [col for col in self.df_procesado.columns 
                       if col not in ['target', 'species']]
        
        stats = self.df_procesado[feature_cols].describe()
        
        return {
            'estadisticas': stats.to_dict(),
            'dataframe': stats
        }


def procesar_iris_completo():
    """
    Función principal que ejecuta todo el procesamiento del dataset Iris
    """
    processor = IrisProcessor()
    
    # 1. Cargar datos
    info_carga = processor.cargar_datos()
    
    # 2. Convertir a DataFrame
    info_dataframe = processor.convertir_a_dataframe()
    
    # 3. Explorar datos
    info_exploracion = processor.explorar_datos()
    
    # 4. Estandarizar
    info_estandarizacion = processor.estandarizar_datos()
    
    # 5. Dividir datos
    info_division = processor.dividir_datos(test_size=0.3, random_state=42)
    
    # 6. Preparar datos para visualización
    datos_visualizacion = processor.obtener_datos_visualizacion()
    
    # 7. Estadísticas descriptivas
    estadisticas = processor.obtener_estadisticas_descriptivas()
    
    # Resumen completo
    resumen = {
        'carga': info_carga,
        'dataframe': info_dataframe,
        'exploracion': info_exploracion,
        'estandarizacion': info_estandarizacion,
        'division': info_division,
        'visualizacion': datos_visualizacion,
        'estadisticas': estadisticas,
        'df_original': processor.df_original,
        'df_procesado': processor.df_procesado,
        'X_train': processor.X_train,
        'X_test': processor.X_test,
        'y_train': processor.y_train,
        'y_test': processor.y_test,
        'target_names': processor.target_names
    }
    
    return processor, resumen
