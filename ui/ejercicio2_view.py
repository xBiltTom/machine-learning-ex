import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ejercicios.ejercicio2.procesamiento import procesar_student_performance_completo


def mostrar_ejercicio2():
    """
    Vista del Ejercicio 2: Dataset Student Performance
    Objetivo: Procesar los datos para un modelo que prediga la nota final (G3) de los estudiantes.
    """
    st.header("📚 Ejercicio 2: Procesamiento del Dataset Student Performance")
    
    st.markdown("""
    ### 📋 Objetivo
    Procesar los datos para un modelo que prediga la **nota final (G3)** de los estudiantes.
    
    ### 🔧 Instrucciones implementadas:
    1. ✅ Carga del dataset y análisis de variables categóricas
    2. ✅ Eliminación de duplicados y valores inconsistentes
    3. ✅ One Hot Encoding a variables categóricas (school, sex, address, etc.)
    4. ✅ Normalización de variables numéricas (age, absences, G1, G2)
    5. ✅ Separación de datos en X y y (características y variable objetivo)
    6. ✅ División en entrenamiento (80%) y prueba (20%)
    
    ### 🎯 Reto adicional:
    - ✅ Análisis de correlación entre las notas G1, G2 y G3
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Ejecutar Procesamiento Completo", type="primary", use_container_width=True):
        with st.spinner("Procesando dataset Student Performance..."):
            try:
                processor, resumen = procesar_student_performance_completo()
                
                st.session_state['student_processor'] = processor
                st.session_state['student_resumen'] = resumen
                st.success("✅ Procesamiento completado exitosamente!")
                
            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {str(e)}")
                return
    
    if 'student_resumen' in st.session_state:
        resumen = st.session_state['student_resumen']
        processor = st.session_state['student_processor']
        
        # 1. CARGA Y EXPLORACIÓN
        st.markdown("## 📥 1. Carga y Análisis de Variables")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", resumen['carga']['filas'])
        with col2:
            st.metric("Columnas Totales", resumen['carga']['columnas'])
        with col3:
            st.metric("Variables Categóricas", len(resumen['carga']['columnas_categoricas']))
        with col4:
            st.metric("Variables Numéricas", len(resumen['carga']['columnas_numericas']))
        
        with st.expander("📊 Ver Dataset Original (primeros 5 registros)"):
            st.dataframe(resumen['df_original'], use_container_width=True)
        
        with st.expander("🔍 Análisis de Variables Categóricas"):
            st.markdown("**Variables categóricas identificadas:**")
            cat_info = []
            for col in resumen['carga']['columnas_categoricas']:
                if col in resumen['exploracion']['valores_unicos_categoricas']:
                    info = resumen['exploracion']['valores_unicos_categoricas'][col]
                    cat_info.append({
                        'Variable': col,
                        'Valores Únicos': info['cantidad'],
                        'Ejemplos': ', '.join(str(v) for v in info['valores'][:5])
                    })
            st.dataframe(pd.DataFrame(cat_info), use_container_width=True)
        
        # 2. LIMPIEZA DE DATOS
        st.markdown("---")
        st.markdown("## 🧹 2. Limpieza de Datos")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duplicados Eliminados", resumen['limpieza']['duplicados_eliminados'])
        with col2:
            st.metric("Valores Nulos Eliminados", resumen['limpieza']['nulos_eliminados'])
        with col3:
            st.metric("Inconsistencias Eliminadas", resumen['limpieza']['inconsistencias_eliminadas'])
        
        st.info(f"📊 Total de filas eliminadas: **{resumen['limpieza']['total_eliminadas']}** "
                f"({resumen['limpieza']['filas_antes']} → {resumen['limpieza']['filas_despues']} registros)")
        
        # 3. ONE HOT ENCODING
        st.markdown("---")
        st.markdown("## 🔢 3. One Hot Encoding de Variables Categóricas")
        
        st.markdown(f"""
        Se aplicó **One Hot Encoding** a las variables categóricas, transformándolas en variables binarias.
        
        - **Columnas antes:** {resumen['codificacion']['total_columnas_antes']}
        - **Columnas después:** {resumen['codificacion']['total_columnas_despues']}
        - **Columnas agregadas:** {resumen['codificacion']['columnas_agregadas']}
        """)
        
        with st.expander("📋 Ver variables categóricas codificadas"):
            st.markdown("**Variables originales codificadas:**")
            for col, info in resumen['codificacion']['valores_por_columna'].items():
                st.write(f"**{col}** ({info['cantidad']} valores únicos): {', '.join(info['valores_unicos'])}")
        
        with st.expander("📋 Ver nuevas columnas generadas (muestra)"):
            nuevas_cols = resumen['codificacion']['columnas_nuevas_creadas'][:20]
            st.write(f"Mostrando 20 de {len(resumen['codificacion']['columnas_nuevas_creadas'])} columnas nuevas:")
            for i, col in enumerate(nuevas_cols, 1):
                st.write(f"{i}. {col}")
        
        # 4. NORMALIZACIÓN
        st.markdown("---")
        st.markdown("## ⚖️ 4. Normalización de Variables Numéricas")
        
        st.markdown("""
        Se aplicó **MinMaxScaler** para normalizar las variables numéricas al rango [0, 1].
        """)
        
        cols_norm = resumen['normalizacion']['columnas'][:4]
        
        if cols_norm:
            col1, col2 = st.columns(2)
            
            stats_antes = []
            stats_despues = []
            
            for col in cols_norm:
                if col in resumen['normalizacion']['estadisticas_antes']:
                    antes = resumen['normalizacion']['estadisticas_antes'][col]
                    despues = resumen['normalizacion']['estadisticas_despues'][col]
                    
                    stats_antes.append({
                        'Variable': col,
                        'Min': f"{antes['min']:.2f}",
                        'Max': f"{antes['max']:.2f}",
                        'Media': f"{antes['media']:.2f}"
                    })
                    
                    stats_despues.append({
                        'Variable': col,
                        'Min': f"{despues['min']:.4f}",
                        'Max': f"{despues['max']:.4f}",
                        'Media': f"{despues['media']:.4f}"
                    })
            
            with col1:
                st.markdown("**📊 Antes de la Normalización**")
                st.dataframe(pd.DataFrame(stats_antes), use_container_width=True)
            
            with col2:
                st.markdown("**📊 Después de la Normalización**")
                st.dataframe(pd.DataFrame(stats_despues), use_container_width=True)
        
        # 5. ANÁLISIS DE CORRELACIÓN (RETO ADICIONAL)
        st.markdown("---")
        st.markdown("## 📈 Reto Adicional: Correlación entre G1, G2 y G3")
        
        if resumen['correlacion']:
            corr_pares = resumen['correlacion']['correlaciones_pares']
            
            col1, col2, col3 = st.columns(3)
            
            if 'G1_G2' in corr_pares:
                with col1:
                    st.metric("Correlación G1 - G2", f"{corr_pares['G1_G2']:.4f}")
            
            if 'G1_G3' in corr_pares:
                with col2:
                    st.metric("Correlación G1 - G3", f"{corr_pares['G1_G3']:.4f}")
            
            if 'G2_G3' in corr_pares:
                with col3:
                    st.metric("Correlación G2 - G3", f"{corr_pares['G2_G3']:.4f}")
            
            # Visualización de matriz de correlación
            st.markdown("### 🔥 Matriz de Correlación (Heatmap)")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            notas_df = resumen['df_original_completo'][['G1', 'G2', 'G3']]
            corr_matrix = notas_df.corr()
            
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0, square=True, linewidths=1, 
                       cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlación entre Notas G1, G2 y G3', fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
            
            st.markdown("""
            **💡 Interpretación:**
            - Valores cercanos a **1**: Correlación positiva fuerte
            - Valores cercanos a **0**: Sin correlación
            - Valores cercanos a **-1**: Correlación negativa fuerte
            """)
            
            # Gráfico de dispersión
            st.markdown("### 📊 Gráficos de Dispersión")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # G1 vs G2
            axes[0].scatter(resumen['df_original_completo']['G1'], 
                          resumen['df_original_completo']['G2'], 
                          alpha=0.5, color='#3498db')
            axes[0].set_xlabel('G1 (Primera nota)')
            axes[0].set_ylabel('G2 (Segunda nota)')
            axes[0].set_title('G1 vs G2')
            axes[0].grid(True, alpha=0.3)
            
            # G1 vs G3
            axes[1].scatter(resumen['df_original_completo']['G1'], 
                          resumen['df_original_completo']['G3'], 
                          alpha=0.5, color='#e74c3c')
            axes[1].set_xlabel('G1 (Primera nota)')
            axes[1].set_ylabel('G3 (Nota final)')
            axes[1].set_title('G1 vs G3')
            axes[1].grid(True, alpha=0.3)
            
            # G2 vs G3
            axes[2].scatter(resumen['df_original_completo']['G2'], 
                          resumen['df_original_completo']['G3'], 
                          alpha=0.5, color='#2ecc71')
            axes[2].set_xlabel('G2 (Segunda nota)')
            axes[2].set_ylabel('G3 (Nota final)')
            axes[2].set_title('G2 vs G3')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # 6. SEPARACIÓN Y DIVISIÓN DE DATOS
        st.markdown("---")
        st.markdown("## ✂️ 5 y 6. Separación de Variables y División de Datos")
        
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
                "Variable objetivo (G3)",
                f"{resumen['division']['num_caracteristicas']} características",
                "Variable objetivo (G3)"
            ]
        }
        
        st.dataframe(pd.DataFrame(dimensiones_data), use_container_width=True)
        
        # Estadísticas de la variable objetivo
        st.markdown("### 📊 Estadísticas de G3 (Variable Objetivo)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Conjunto de Entrenamiento**")
            stats_train = resumen['division']['estadisticas_y']['train']
            stats_train_df = pd.DataFrame({
                'Métrica': ['Media', 'Desviación Est.', 'Mínimo', 'Máximo'],
                'Valor': [
                    f"{stats_train['media']:.2f}",
                    f"{stats_train['std']:.2f}",
                    f"{stats_train['min']:.2f}",
                    f"{stats_train['max']:.2f}"
                ]
            })
            st.dataframe(stats_train_df, use_container_width=True)
        
        with col2:
            st.markdown("**Conjunto de Prueba**")
            stats_test = resumen['division']['estadisticas_y']['test']
            stats_test_df = pd.DataFrame({
                'Métrica': ['Media', 'Desviación Est.', 'Mínimo', 'Máximo'],
                'Valor': [
                    f"{stats_test['media']:.2f}",
                    f"{stats_test['std']:.2f}",
                    f"{stats_test['min']:.2f}",
                    f"{stats_test['max']:.2f}"
                ]
            })
            st.dataframe(stats_test_df, use_container_width=True)
        
        # 7. RESULTADOS FINALES
        st.markdown("---")
        st.markdown("## 🎯 Resultados Finales")
        
        st.markdown("### 📋 Dataset Procesado (primeros 5 registros)")
        st.dataframe(resumen['df_procesado'], use_container_width=True)
        
        # Visualización final
        st.markdown("### 📈 Visualización de Distribuciones")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Análisis del Dataset Student Performance', fontsize=16, fontweight='bold')
        
        # Distribución de G3
        axes[0, 0].hist(resumen['df_original_completo']['G3'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribución de G3 (Nota Final)')
        axes[0, 0].set_xlabel('Nota G3')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Distribución de G2
        axes[0, 1].hist(resumen['df_original_completo']['G2'], bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribución de G2 (Segunda Nota)')
        axes[0, 1].set_xlabel('Nota G2')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Distribución de G1
        axes[1, 0].hist(resumen['df_original_completo']['G1'], bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribución de G1 (Primera Nota)')
        axes[1, 0].set_xlabel('Nota G1')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Distribución de Age
        axes[1, 1].hist(resumen['df_original_completo']['age'], bins=15, color='#9b59b6', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribución de Edad')
        axes[1, 1].set_xlabel('Edad')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Descarga
        st.markdown("---")
        st.markdown("### 💾 Descargar Datos Procesados")
        
        csv = resumen['df_procesado'].to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV Procesado",
            data=csv,
            file_name="student_performance_procesado.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("👆 Haz clic en el botón **'Ejecutar Procesamiento Completo'** para comenzar el análisis.")
