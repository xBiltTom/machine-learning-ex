import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Agregar el path para importar el módulo de procesamiento
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ejercicios.ejercicio1.procesamiento import procesar_titanic_completo


def mostrar_ejercicio1():
    """
    Vista del Ejercicio 1: Dataset Titanic
    Objetivo: Preparar los datos para un modelo que prediga la supervivencia de los pasajeros.
    """
    st.header("🚢 Ejercicio 1: Análisis del Dataset Titanic")
    
    st.markdown("""
    ### 📋 Objetivo
    Preparar los datos para un modelo que prediga la **supervivencia de los pasajeros** del Titanic.
    
    ### 🔧 Instrucciones implementadas:
    1. ✅ Carga del dataset con pandas
    2. ✅ Eliminación de columnas irrelevantes (Name, Ticket, Cabin, PassengerId)
    3. ✅ Verificación y reemplazo de valores nulos (media/moda)
    4. ✅ Codificación de variables Sex y Embarked
    5. ✅ Estandarización de variables numéricas (Age, Fare)
    6. ✅ División en entrenamiento (70%) y prueba (30%)
    """)
    
    st.markdown("---")
    
    # Botón para ejecutar el procesamiento
    if st.button("🔄 Ejecutar Procesamiento Completo", type="primary", use_container_width=True):
        with st.spinner("Procesando dataset del Titanic..."):
            try:
                # Ejecutar procesamiento
                processor, resumen = procesar_titanic_completo()
                
                # Guardar en session_state para persistencia
                st.session_state['titanic_processor'] = processor
                st.session_state['titanic_resumen'] = resumen
                st.success("✅ Procesamiento completado exitosamente!")
                
            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {str(e)}")
                return
    
    # Mostrar resultados si existen en session_state
    if 'titanic_resumen' in st.session_state:
        resumen = st.session_state['titanic_resumen']
        processor = st.session_state['titanic_processor']
        
        # ====================
        # 1. CARGA Y EXPLORACIÓN INICIAL
        # ====================
        st.markdown("## 📥 1. Carga y Exploración Inicial")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", resumen['carga']['filas'])
        with col2:
            st.metric("Columnas Originales", resumen['carga']['columnas'])
        with col3:
            nulos_totales = sum(resumen['exploracion']['nulos'].values())
            st.metric("Valores Nulos", nulos_totales)
        with col4:
            st.metric("Duplicados", resumen['exploracion']['duplicados'])
        
        with st.expander("📊 Ver Dataset Original (primeros 5 registros)"):
            st.dataframe(resumen['df_original'], use_container_width=True)
        
        with st.expander("🔍 Información de Valores Nulos (Original)"):
            nulos_df = pd.DataFrame({
                'Columna': list(resumen['exploracion']['nulos'].keys()),
                'Valores Nulos': list(resumen['exploracion']['nulos'].values()),
                'Porcentaje (%)': [f"{v:.2f}%" for v in resumen['exploracion']['nulos_porcentaje'].values()]
            })
            nulos_df = nulos_df[nulos_df['Valores Nulos'] > 0]
            if not nulos_df.empty:
                st.dataframe(nulos_df, use_container_width=True)
            else:
                st.info("No hay valores nulos en el dataset original")
        
        # ====================
        # 2. ELIMINACIÓN DE COLUMNAS
        # ====================
        st.markdown("---")
        st.markdown("## 🗑️ 2. Eliminación de Columnas Irrelevantes")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Columnas eliminadas:**")
            for col in resumen['eliminacion']['eliminadas']:
                st.write(f"- ❌ {col}")
        
        with col2:
            st.markdown("**Columnas restantes:**")
            for col in resumen['eliminacion']['columnas_restantes']:
                st.write(f"- ✅ {col}")
        
        # ====================
        # 3. MANEJO DE VALORES NULOS
        # ====================
        st.markdown("---")
        st.markdown("## 🧹 3. Limpieza de Datos - Valores Nulos")
        
        col1, col2, col3 = st.columns(3)
        
        if 'Age_reemplazo' in resumen['nulos']:
            with col1:
                st.info(f"**Age (Edad)**\n\n{resumen['nulos']['Age_reemplazo']}")
        
        if 'Fare_reemplazo' in resumen['nulos']:
            with col2:
                st.info(f"**Fare (Tarifa)**\n\n{resumen['nulos']['Fare_reemplazo']}")
        
        if 'Embarked_reemplazo' in resumen['nulos']:
            with col3:
                st.info(f"**Embarked (Puerto)**\n\n{resumen['nulos']['Embarked_reemplazo']}")
        
        if resumen['nulos']['filas_eliminadas'] > 0:
            st.warning(f"⚠️ Se eliminaron {resumen['nulos']['filas_eliminadas']} filas con valores nulos restantes")
        
        # ====================
        # 4. CODIFICACIÓN DE VARIABLES
        # ====================
        st.markdown("---")
        st.markdown("## 🔢 4. Codificación de Variables Categóricas")
        
        col1, col2 = st.columns(2)
        
        if 'Sex' in resumen['codificacion']:
            with col1:
                st.markdown("**Variable: Sex (Sexo)**")
                mapeo_sex = resumen['codificacion']['Sex']['mapeo']
                mapeo_df = pd.DataFrame(list(mapeo_sex.items()), columns=['Valor Original', 'Valor Codificado'])
                st.dataframe(mapeo_df, use_container_width=True)
        
        if 'Embarked' in resumen['codificacion']:
            with col2:
                st.markdown("**Variable: Embarked (Puerto de Embarque)**")
                mapeo_embarked = resumen['codificacion']['Embarked']['mapeo']
                mapeo_df = pd.DataFrame(list(mapeo_embarked.items()), columns=['Valor Original', 'Valor Codificado'])
                st.dataframe(mapeo_df, use_container_width=True)
        
        # ====================
        # 5. ESTANDARIZACIÓN
        # ====================
        st.markdown("---")
        st.markdown("## ⚖️ 5. Estandarización de Variables Numéricas")
        
        st.markdown("""
        Se aplicó **StandardScaler** para estandarizar las variables numéricas.
        La estandarización transforma los datos para que tengan **media = 0** y **desviación estándar = 1**.
        """)
        
        # Crear tabla comparativa
        estadisticas_antes = []
        estadisticas_despues = []
        
        for col in resumen['estandarizacion']['columnas']:
            antes = resumen['estandarizacion']['estadisticas_antes'][col]
            despues = resumen['estandarizacion']['estadisticas_despues'][col]
            
            estadisticas_antes.append({
                'Variable': col,
                'Media': f"{antes['media']:.2f}",
                'Desv. Est.': f"{antes['std']:.2f}",
                'Min': f"{antes['min']:.2f}",
                'Max': f"{antes['max']:.2f}"
            })
            
            estadisticas_despues.append({
                'Variable': col,
                'Media': f"{despues['media']:.4f}",
                'Desv. Est.': f"{despues['std']:.4f}",
                'Min': f"{despues['min']:.2f}",
                'Max': f"{despues['max']:.2f}"
            })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Antes de la Estandarización**")
            st.dataframe(pd.DataFrame(estadisticas_antes), use_container_width=True)
        
        with col2:
            st.markdown("**📊 Después de la Estandarización**")
            st.dataframe(pd.DataFrame(estadisticas_despues), use_container_width=True)
        
        # ====================
        # 6. DIVISIÓN DE DATOS
        # ====================
        st.markdown("---")
        st.markdown("## ✂️ 6. División en Conjuntos de Entrenamiento y Prueba")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Registros", resumen['division']['total_registros'])
        
        with col2:
            st.metric(
                "Conjunto de Entrenamiento", 
                f"{resumen['division']['train_shape']['X_train'][0]} registros",
                f"{resumen['division']['porcentajes']['train']}"
            )
        
        with col3:
            st.metric(
                "Conjunto de Prueba", 
                f"{resumen['division']['test_shape']['X_test'][0]} registros",
                f"{resumen['division']['porcentajes']['test']}"
            )
        
        # Mostrar dimensiones detalladas
        st.markdown("### 📐 Dimensiones de los Conjuntos")
        
        dimensiones_data = {
            'Conjunto': ['X_train (Características)', 'y_train (Objetivo)', 'X_test (Características)', 'y_test (Objetivo)'],
            'Dimensiones': [
                str(resumen['division']['train_shape']['X_train']),
                str(resumen['division']['train_shape']['y_train']),
                str(resumen['division']['test_shape']['X_test']),
                str(resumen['division']['test_shape']['y_test'])
            ],
            'Descripción': [
                f"{resumen['division']['train_shape']['X_train'][1]} características",
                "Variable objetivo (Survived)",
                f"{resumen['division']['test_shape']['X_test'][1]} características",
                "Variable objetivo (Survived)"
            ]
        }
        
        st.dataframe(pd.DataFrame(dimensiones_data), use_container_width=True)
        
        # Distribución de clases
        st.markdown("### 📊 Distribución de Clases (Survived)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Entrenamiento**")
            dist_train = resumen['division']['distribucion_clases_train']
            dist_train_df = pd.DataFrame({
                'Clase': ['No Sobrevivió (0)', 'Sobrevivió (1)'],
                'Cantidad': [dist_train.get(0, 0), dist_train.get(1, 0)]
            })
            st.dataframe(dist_train_df, use_container_width=True)
        
        with col2:
            st.markdown("**Prueba**")
            dist_test = resumen['division']['distribucion_clases_test']
            dist_test_df = pd.DataFrame({
                'Clase': ['No Sobrevivió (0)', 'Sobrevivió (1)'],
                'Cantidad': [dist_test.get(0, 0), dist_test.get(1, 0)]
            })
            st.dataframe(dist_test_df, use_container_width=True)
        
        # ====================
        # 7. RESULTADOS FINALES
        # ====================
        st.markdown("---")
        st.markdown("## 🎯 Resultados Finales")
        
        st.markdown("### 📋 Tabla con los Primeros 5 Registros Procesados")
        st.dataframe(resumen['df_procesado'], use_container_width=True)
        
        # Características finales
        st.markdown("### 🔧 Características Finales del Dataset")
        caracteristicas_info = {
            'Característica': resumen['division']['caracteristicas'],
            'Tipo': ['Numérica' if col in ['Age', 'Fare'] else 'Categórica' for col in resumen['division']['caracteristicas']]
        }
        st.dataframe(pd.DataFrame(caracteristicas_info), use_container_width=True)
        
        # Visualización de distribuciones
        st.markdown("### 📈 Visualización de Datos Procesados")
        
        # Crear visualización
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Análisis del Dataset Titanic Procesado', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Distribución de Survived
        survived_counts = resumen['df_procesado']['Survived'].value_counts()
        axes[0, 0].bar(['No Sobrevivió (0)', 'Sobrevivió (1)'], survived_counts.values, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_title('Distribución de Supervivencia')
        axes[0, 0].set_ylabel('Cantidad')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Gráfico 2: Distribución de Pclass
        pclass_counts = resumen['df_procesado']['Pclass'].value_counts().sort_index()
        axes[0, 1].bar(pclass_counts.index, pclass_counts.values, color=['#95E1D3', '#F38181', '#AA96DA'])
        axes[0, 1].set_title('Distribución por Clase (Pclass)')
        axes[0, 1].set_xlabel('Clase')
        axes[0, 1].set_ylabel('Cantidad')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Gráfico 3: Distribución de Age (estandarizada)
        axes[1, 0].hist(resumen['df_procesado']['Age'], bins=30, color='#6C5CE7', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribución de Age (Estandarizada)')
        axes[1, 0].set_xlabel('Age (Estandarizada)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Gráfico 4: Distribución de Fare (estandarizada)
        axes[1, 1].hist(resumen['df_procesado']['Fare'], bins=30, color='#FD79A8', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribución de Fare (Estandarizada)')
        axes[1, 1].set_xlabel('Fare (Estandarizada)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Botón de descarga
        st.markdown("---")
        st.markdown("### 💾 Descargar Datos Procesados")
        
        csv = resumen['df_procesado'].to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV Procesado",
            data=csv,
            file_name="titanic_procesado.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("👆 Haz clic en el botón **'Ejecutar Procesamiento Completo'** para comenzar el análisis del dataset Titanic.")
