"""
Ejercicio 2: Procesamiento del Dataset Student Performance
Objetivo: Procesar los datos para un modelo que prediga la nota final (G3) de los estudiantes.

Etapas:
1. Carga del dataset y análisis de variables categóricas
2. Eliminación de duplicados y valores inconsistentes
3. One Hot Encoding a variables categóricas
4. Normalización de variables numéricas
5. Separación de datos en X y y
6. División en conjuntos de entrenamiento y prueba
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


class StudentPerformanceProcessor:
    """Clase para procesar el dataset Student Performance"""
    
    def __init__(self):
        self.df_original = None
        self.df_procesado = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = MinMaxScaler()
        self.columnas_categoricas = []
        self.columnas_numericas = []
        self.info_duplicados = {}
        self.info_codificacion = {}
        
    def cargar_datos(self):
        """
        1. Carga del dataset y análisis de variables categóricas
        """
        ruta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_csv = os.path.join(ruta_actual, 'student-mat.csv')
        
        self.df_original = pd.read_csv(ruta_csv)
        self.df_procesado = self.df_original.copy()
        
        # Identificar variables categóricas y numéricas
        self.columnas_categoricas = self.df_procesado.select_dtypes(include=['object']).columns.tolist()
        self.columnas_numericas = self.df_procesado.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        return {
            'filas': self.df_original.shape[0],
            'columnas': self.df_original.shape[1],
            'columnas_nombres': list(self.df_original.columns),
            'columnas_categoricas': self.columnas_categoricas,
            'columnas_numericas': self.columnas_numericas,
            'tipos_datos': self.df_original.dtypes.to_dict()
        }
    
    def explorar_datos(self):
        """
        Exploración inicial del dataset
        """
        info = {
            'shape': self.df_original.shape,
            'columnas': list(self.df_original.columns),
            'tipos': self.df_original.dtypes.to_dict(),
            'nulos': self.df_original.isnull().sum().to_dict(),
            'nulos_porcentaje': (self.df_original.isnull().sum() / len(self.df_original) * 100).to_dict(),
            'duplicados': self.df_original.duplicated().sum(),
            'estadisticas': self.df_original.describe().to_dict(),
            'valores_unicos_categoricas': {}
        }
        
        # Analizar valores únicos de variables categóricas
        for col in self.columnas_categoricas:
            info['valores_unicos_categoricas'][col] = {
                'cantidad': self.df_original[col].nunique(),
                'valores': self.df_original[col].unique().tolist()
            }
        
        return info
    
    def eliminar_duplicados_e_inconsistencias(self):
        """
        2. Elimine duplicados y valores inconsistentes
        """
        filas_antes = len(self.df_procesado)
        
        # Eliminar duplicados
        duplicados_antes = self.df_procesado.duplicated().sum()
        self.df_procesado = self.df_procesado.drop_duplicates()
        duplicados_eliminados = duplicados_antes
        
        # Eliminar filas con valores nulos (si existen)
        nulos_antes = self.df_procesado.isnull().sum().sum()
        self.df_procesado = self.df_procesado.dropna()
        nulos_eliminados = nulos_antes
        
        # Validar rangos de notas G1, G2, G3 (deben estar entre 0 y 20)
        inconsistentes = 0
        for col in ['G1', 'G2', 'G3']:
            if col in self.df_procesado.columns:
                mask = (self.df_procesado[col] >= 0) & (self.df_procesado[col] <= 20)
                inconsistentes += (~mask).sum()
                self.df_procesado = self.df_procesado[mask]
        
        filas_despues = len(self.df_procesado)
        
        self.info_duplicados = {
            'filas_antes': filas_antes,
            'filas_despues': filas_despues,
            'duplicados_eliminados': duplicados_eliminados,
            'nulos_eliminados': nulos_eliminados,
            'inconsistencias_eliminadas': inconsistentes,
            'total_eliminadas': filas_antes - filas_despues
        }
        
        return self.info_duplicados
    
    def aplicar_one_hot_encoding(self):
        """
        3. Aplique One Hot Encoding a las variables categóricas
        """
        columnas_antes = list(self.df_procesado.columns)
        categoricas_a_codificar = [col for col in self.columnas_categoricas if col in self.df_procesado.columns]
        
        self.info_codificacion = {
            'columnas_originales': categoricas_a_codificar,
            'valores_por_columna': {}
        }
        
        # Guardar información antes de codificar
        for col in categoricas_a_codificar:
            self.info_codificacion['valores_por_columna'][col] = {
                'valores_unicos': self.df_procesado[col].unique().tolist(),
                'cantidad': self.df_procesado[col].nunique()
            }
        
        # Aplicar One Hot Encoding
        self.df_procesado = pd.get_dummies(self.df_procesado, columns=categoricas_a_codificar, drop_first=False)
        
        columnas_despues = list(self.df_procesado.columns)
        columnas_nuevas = [col for col in columnas_despues if col not in columnas_antes]
        
        self.info_codificacion['columnas_despues'] = columnas_despues
        self.info_codificacion['columnas_nuevas_creadas'] = columnas_nuevas
        self.info_codificacion['total_columnas_antes'] = len(columnas_antes)
        self.info_codificacion['total_columnas_despues'] = len(columnas_despues)
        self.info_codificacion['columnas_agregadas'] = len(columnas_despues) - len(columnas_antes)
        
        return self.info_codificacion
    
    def normalizar_variables(self):
        """
        4. Normalice las variables numéricas (age, absences, G1, G2)
        Nota: G3 es la variable objetivo, no se normaliza en este paso
        """
        # Variables a normalizar (excluyendo G3 que es la variable objetivo)
        columnas_a_normalizar = ['age', 'absences', 'G1', 'G2']
        columnas_disponibles = [col for col in columnas_a_normalizar if col in self.df_procesado.columns]
        
        # También normalizar otras variables numéricas que no sean G3
        otras_numericas = [col for col in self.df_procesado.columns 
                          if col in self.columnas_numericas 
                          and col not in columnas_a_normalizar 
                          and col != 'G3']
        
        todas_a_normalizar = columnas_disponibles + otras_numericas
        
        info_normalizacion = {
            'columnas': todas_a_normalizar,
            'estadisticas_antes': {},
            'estadisticas_despues': {}
        }
        
        # Guardar estadísticas antes de normalizar
        for col in todas_a_normalizar:
            info_normalizacion['estadisticas_antes'][col] = {
                'media': float(self.df_procesado[col].mean()),
                'std': float(self.df_procesado[col].std()),
                'min': float(self.df_procesado[col].min()),
                'max': float(self.df_procesado[col].max())
            }
        
        # Normalizar usando MinMaxScaler (0-1)
        if todas_a_normalizar:
            self.df_procesado[todas_a_normalizar] = self.scaler.fit_transform(
                self.df_procesado[todas_a_normalizar]
            )
        
        # Guardar estadísticas después de normalizar
        for col in todas_a_normalizar:
            info_normalizacion['estadisticas_despues'][col] = {
                'media': float(self.df_procesado[col].mean()),
                'std': float(self.df_procesado[col].std()),
                'min': float(self.df_procesado[col].min()),
                'max': float(self.df_procesado[col].max())
            }
        
        return info_normalizacion
    
    def analizar_correlacion_notas(self):
        """
        Reto adicional: Analice la correlación entre las notas G1, G2 y G3
        """
        notas = ['G1', 'G2', 'G3']
        notas_disponibles = [col for col in notas if col in self.df_original.columns]
        
        if len(notas_disponibles) < 2:
            return None
        
        # Calcular matriz de correlación
        correlacion = self.df_original[notas_disponibles].corr()
        
        info_correlacion = {
            'matriz_correlacion': correlacion.to_dict(),
            'correlaciones_pares': {}
        }
        
        # Extraer correlaciones específicas
        if 'G1' in notas_disponibles and 'G2' in notas_disponibles:
            info_correlacion['correlaciones_pares']['G1_G2'] = float(correlacion.loc['G1', 'G2'])
        
        if 'G1' in notas_disponibles and 'G3' in notas_disponibles:
            info_correlacion['correlaciones_pares']['G1_G3'] = float(correlacion.loc['G1', 'G3'])
        
        if 'G2' in notas_disponibles and 'G3' in notas_disponibles:
            info_correlacion['correlaciones_pares']['G2_G3'] = float(correlacion.loc['G2', 'G3'])
        
        return info_correlacion
    
    def separar_y_dividir_datos(self, test_size=0.2, random_state=42):
        """
        5 y 6. Separe los datos en X y y, y divida en entrenamiento (80%) y prueba (20%)
        """
        # Verificar que G3 existe
        if 'G3' not in self.df_procesado.columns:
            raise ValueError("La variable objetivo G3 no está en el dataset")
        
        # Separar características (X) y variable objetivo (y)
        X = self.df_procesado.drop('G3', axis=1)
        y = self.df_procesado['G3']
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        info_division = {
            'total_registros': len(self.df_procesado),
            'caracteristicas': list(X.columns),
            'num_caracteristicas': len(X.columns),
            'variable_objetivo': 'G3',
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
            'estadisticas_y': {
                'train': {
                    'media': float(self.y_train.mean()),
                    'std': float(self.y_train.std()),
                    'min': float(self.y_train.min()),
                    'max': float(self.y_train.max())
                },
                'test': {
                    'media': float(self.y_test.mean()),
                    'std': float(self.y_test.std()),
                    'min': float(self.y_test.min()),
                    'max': float(self.y_test.max())
                }
            }
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


def procesar_student_performance_completo():
    """
    Función principal que ejecuta todo el procesamiento del dataset Student Performance
    """
    processor = StudentPerformanceProcessor()
    
    # 1. Cargar datos
    info_carga = processor.cargar_datos()
    
    # 2. Explorar datos
    info_exploracion = processor.explorar_datos()
    
    # 3. Eliminar duplicados e inconsistencias
    info_limpieza = processor.eliminar_duplicados_e_inconsistencias()
    
    # 4. Aplicar One Hot Encoding
    info_codificacion = processor.aplicar_one_hot_encoding()
    
    # 5. Normalizar variables numéricas
    info_normalizacion = processor.normalizar_variables()
    
    # 6. Analizar correlación (reto adicional)
    info_correlacion = processor.analizar_correlacion_notas()
    
    # 7. Separar y dividir datos
    info_division = processor.separar_y_dividir_datos(test_size=0.2, random_state=42)
    
    # Resumen completo
    resumen = {
        'carga': info_carga,
        'exploracion': info_exploracion,
        'limpieza': info_limpieza,
        'codificacion': info_codificacion,
        'normalizacion': info_normalizacion,
        'correlacion': info_correlacion,
        'division': info_division,
        'df_original': processor.obtener_dataframe_original(5),
        'df_procesado': processor.obtener_primeros_registros(5),
        'X_train': processor.X_train,
        'X_test': processor.X_test,
        'y_train': processor.y_train,
        'y_test': processor.y_test,
        'df_original_completo': processor.df_original
    }
    
    return processor, resumen
