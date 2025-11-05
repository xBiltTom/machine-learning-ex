import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ejercicios.ejercicio3.procesamiento import procesar_iris_completo


def mostrar_ejercicio3():
    """
    Vista del Ejercicio 3: Dataset Iris
    Objetivo: Implementar un flujo completo de preprocesamiento y visualizar resultados.
    """
    st.header("🌸 Ejercicio 3: Dataset Iris")
    
    st.markdown("""
    ### 📋 Objetivo
    Implementar un **flujo completo de preprocesamiento** y visualizar resultados del dataset Iris.
    
    ### 🔧 Instrucciones implementadas:
    1. ✅ Carga del dataset desde sklearn.datasets
    2. ✅ Conversión a DataFrame con nombres de columnas
    3. ✅ Estandarización con StandardScaler()
    4. ✅ División del dataset (70% entrenamiento, 30% prueba)
    5. ✅ Gráfico de dispersión (sepal length vs petal length) por clase
    
    ### 📈 Salidas esperadas:
    - Gráfico de dispersión con colores por clase
    - Estadísticas descriptivas del dataset estandarizado
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Ejecutar Procesamiento Completo", type="primary", use_container_width=True):
        with st.spinner("Procesando dataset Iris..."):
            try:
                processor, resumen = procesar_iris_completo()
                
                st.session_state['iris_processor'] = processor
                st.session_state['iris_resumen'] = resumen
                st.success("✅ Procesamiento completado exitosamente!")
                
            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {str(e)}")
                return
    
    if 'iris_resumen' in st.session_state:
        resumen = st.session_state['iris_resumen']
        processor = st.session_state['iris_processor']
        
        # 1. CARGA DEL DATASET
        st.markdown("## 📥 1. Carga del Dataset desde scikit-learn")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Muestras", resumen['carga']['n_samples'])
        with col2:
            st.metric("Características", resumen['carga']['n_features'])
        with col3:
            st.metric("Clases", len(resumen['carga']['target_names']))
        with col4:
            st.metric("Dataset", "Iris")
        
        with st.expander("ℹ️ Información del Dataset"):
            st.markdown("**Características del dataset:**")
            for i, feature in enumerate(resumen['carga']['feature_names'], 1):
                st.write(f"{i}. {feature}")
            
            st.markdown("**Clases (especies):**")
            for i, target in enumerate(resumen['carga']['target_names'], 1):
                st.write(f"{i}. {target}")
        
        # 2. CONVERSIÓN A DATAFRAME
        st.markdown("---")
        st.markdown("## 📊 2. Conversión a DataFrame")
        
        st.markdown(f"Dataset convertido a DataFrame con **{resumen['dataframe']['shape'][0]} filas** y **{resumen['dataframe']['shape'][1]} columnas**")
        
        with st.expander("📋 Ver Dataset Original (primeros 10 registros)"):
            st.dataframe(resumen['df_original'].head(10), use_container_width=True)
        
        # Distribución de clases
        st.markdown("### 📊 Distribución de Clases")
        
        cols = st.columns(len(resumen['exploracion']['info_clases']))
        for i, (especie, info) in enumerate(resumen['exploracion']['info_clases'].items()):
            with cols[i]:
                st.metric(especie.capitalize(), info['cantidad'], info['porcentaje'])
        
        # 3. ESTANDARIZACIÓN
        st.markdown("---")
        st.markdown("## ⚖️ 3. Estandarización con StandardScaler")
        
        st.markdown("""
        Se aplicó **StandardScaler** para estandarizar las características.
        La estandarización transforma los datos para que tengan **media = 0** y **desviación estándar = 1**.
        """)
        
        # Comparación antes y después
        col1, col2 = st.columns(2)
        
        stats_antes = []
        stats_despues = []
        
        for col in resumen['estandarizacion']['columnas_estandarizadas']:
            antes = resumen['estandarizacion']['estadisticas_antes'][col]
            despues = resumen['estandarizacion']['estadisticas_despues'][col]
            
            # Nombre simplificado
            nombre_corto = col.replace(' (cm)', '').title()
            
            stats_antes.append({
                'Característica': nombre_corto,
                'Media': f"{antes['media']:.2f}",
                'Desv. Est.': f"{antes['std']:.2f}",
                'Min': f"{antes['min']:.2f}",
                'Max': f"{antes['max']:.2f}"
            })
            
            stats_despues.append({
                'Característica': nombre_corto,
                'Media': f"{despues['media']:.4f}",
                'Desv. Est.': f"{despues['std']:.4f}",
                'Min': f"{despues['min']:.2f}",
                'Max': f"{despues['max']:.2f}"
            })
        
        with col1:
            st.markdown("**📊 Antes de la Estandarización**")
            st.dataframe(pd.DataFrame(stats_antes), use_container_width=True)
        
        with col2:
            st.markdown("**� Después de la Estandarización**")
            st.dataframe(pd.DataFrame(stats_despues), use_container_width=True)
        
        # 4. DIVISIÓN DE DATOS
        st.markdown("---")
        st.markdown("## ✂️ 4. División del Dataset")
        
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
        
        # Dimensiones
        st.markdown("### 📐 Dimensiones de los Conjuntos")
        
        dimensiones_data = {
            'Conjunto': ['X_train', 'y_train', 'X_test', 'y_test'],
            'Dimensiones': [
                str(resumen['division']['train_shape']['X_train']),
                str(resumen['division']['train_shape']['y_train']),
                str(resumen['division']['test_shape']['X_test']),
                str(resumen['division']['test_shape']['y_test'])
            ],
            'Descripción': [
                f"{resumen['division']['num_caracteristicas']} características",
                "Variable objetivo (target)",
                f"{resumen['division']['num_caracteristicas']} características",
                "Variable objetivo (target)"
            ]
        }
        
        st.dataframe(pd.DataFrame(dimensiones_data), use_container_width=True)
        
        # Distribución por conjunto
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribución en Entrenamiento**")
            dist_train = resumen['division']['distribucion_clases_train']
            dist_train_df = pd.DataFrame({
                'Clase': [resumen['target_names'][k] for k in sorted(dist_train.keys())],
                'Cantidad': [dist_train[k] for k in sorted(dist_train.keys())]
            })
            st.dataframe(dist_train_df, use_container_width=True)
        
        with col2:
            st.markdown("**Distribución en Prueba**")
            dist_test = resumen['division']['distribucion_clases_test']
            dist_test_df = pd.DataFrame({
                'Clase': [resumen['target_names'][k] for k in sorted(dist_test.keys())],
                'Cantidad': [dist_test[k] for k in sorted(dist_test.keys())]
            })
            st.dataframe(dist_test_df, use_container_width=True)
        
        # 5. VISUALIZACIÓN - GRÁFICO DE DISPERSIÓN
        st.markdown("---")
        st.markdown("## 📈 5. Visualización: Sepal Length vs Petal Length por Clase")
        
        # Colores para cada clase
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        nombres_especies = resumen['target_names']
        
        # Crear gráficos lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Datos Originales**")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            
            viz_original = resumen['visualizacion']['original']
            
            for i, especie in enumerate(nombres_especies):
                mask = [target == i for target in viz_original['target']]
                sepal_vals = [viz_original['sepal_length'][j] for j in range(len(mask)) if mask[j]]
                petal_vals = [viz_original['petal_length'][j] for j in range(len(mask)) if mask[j]]
                
                ax1.scatter(sepal_vals, petal_vals, 
                          c=colores[i], label=especie.capitalize(), 
                          alpha=0.6, edgecolors='black', s=80)
            
            ax1.set_xlabel('Sepal Length (cm)', fontsize=12)
            ax1.set_ylabel('Petal Length (cm)', fontsize=12)
            ax1.set_title('Distribución Original por Clase', fontsize=14, fontweight='bold')
            ax1.legend(title='Especies')
            ax1.grid(True, alpha=0.3)
            
            st.pyplot(fig1)
        
        with col2:
            st.markdown("**📊 Datos Estandarizados**")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            
            viz_estandarizado = resumen['visualizacion']['estandarizado']
            
            for i, especie in enumerate(nombres_especies):
                mask = [target == i for target in viz_estandarizado['target']]
                sepal_vals = [viz_estandarizado['sepal_length'][j] for j in range(len(mask)) if mask[j]]
                petal_vals = [viz_estandarizado['petal_length'][j] for j in range(len(mask)) if mask[j]]
                
                ax2.scatter(sepal_vals, petal_vals, 
                          c=colores[i], label=especie.capitalize(), 
                          alpha=0.6, edgecolors='black', s=80)
            
            ax2.set_xlabel('Sepal Length (Estandarizada)', fontsize=12)
            ax2.set_ylabel('Petal Length (Estandarizada)', fontsize=12)
            ax2.set_title('Distribución Estandarizada por Clase', fontsize=14, fontweight='bold')
            ax2.legend(title='Especies')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig2)
        
        # 6. ESTADÍSTICAS DESCRIPTIVAS
        st.markdown("---")
        st.markdown("## 📊 Estadísticas Descriptivas del Dataset Estandarizado")
        
        st.dataframe(resumen['estadisticas']['dataframe'], use_container_width=True)
        
        # Visualizaciones adicionales
        st.markdown("### 📈 Visualizaciones Adicionales")
        
        # Gráfico de todas las características
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribución de Todas las Características por Clase', fontsize=16, fontweight='bold')
        
        feature_names = resumen['division']['caracteristicas']
        
        for idx, feature in enumerate(feature_names):
            row = idx // 2
            col = idx % 2
            
            for i, especie in enumerate(nombres_especies):
                mask = resumen['df_procesado']['target'] == i
                data = resumen['df_procesado'][mask][feature]
                axes[row, col].hist(data, bins=15, alpha=0.5, label=especie.capitalize(), 
                                   color=colores[i], edgecolor='black')
            
            feature_label = feature.replace(' (cm)', '').title()
            axes[row, col].set_xlabel(feature_label, fontsize=10)
            axes[row, col].set_ylabel('Frecuencia', fontsize=10)
            axes[row, col].set_title(f'Distribución de {feature_label}', fontsize=12)
            axes[row, col].legend()
            axes[row, col].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Matriz de correlación
        st.markdown("### 🔥 Matriz de Correlación de Características")
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        
        feature_cols = [col for col in resumen['df_procesado'].columns 
                       if col not in ['target', 'species']]
        corr_matrix = resumen['df_procesado'][feature_cols].corr()
        
        # Nombres simplificados para la matriz
        labels_cortos = [col.replace(' (cm)', '').replace('sepal', 'Sep').replace('petal', 'Pet') 
                        for col in feature_cols]
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1,
                   xticklabels=labels_cortos, yticklabels=labels_cortos,
                   cbar_kws={"shrink": 0.8}, ax=ax_corr)
        ax_corr.set_title('Matriz de Correlación - Dataset Iris', fontsize=14, fontweight='bold')
        
        st.pyplot(fig_corr)
        
        # Descarga
        st.markdown("---")
        st.markdown("### 💾 Descargar Datos Procesados")
        
        csv = resumen['df_procesado'].to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV Procesado",
            data=csv,
            file_name="iris_procesado.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("👆 Haz clic en el botón **'Ejecutar Procesamiento Completo'** para comenzar el análisis.")
