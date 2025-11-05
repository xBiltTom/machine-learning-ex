import streamlit as st

# Importar las vistas de los ejercicios
from ui.ejercicio1_view import mostrar_ejercicio1
from ui.ejercicio2_view import mostrar_ejercicio2
from ui.ejercicio3_view import mostrar_ejercicio3

# Configuración de la página
st.set_page_config(
    page_title="Machine Learning - Procesamiento de Datasets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Procesamiento de Datasets en Machine Learning")
st.markdown("---")

# Sidebar con navegación
with st.sidebar:
    st.header("🔍 Navegación")
    st.markdown("Selecciona el ejercicio que deseas visualizar:")
    
    ejercicio_seleccionado = st.radio(
        "Ejercicios disponibles:",
        ["🏠 Inicio", "🚢 Ejercicio 1: Titanic", "📚 Ejercicio 2: Student Performance", "🌸 Ejercicio 3: Iris"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📋 Descripción")
    
    if ejercicio_seleccionado == "🏠 Inicio":
        st.info("Selecciona un ejercicio del menú para ver su implementación.")
    elif ejercicio_seleccionado == "🚢 Ejercicio 1: Titanic":
        st.info("**Objetivo:** Preparar datos para predecir la supervivencia de pasajeros del Titanic.")
    elif ejercicio_seleccionado == "📚 Ejercicio 2: Student Performance":
        st.info("**Objetivo:** Procesar datos para predecir la nota final (G3) de estudiantes.")
    elif ejercicio_seleccionado == "🌸 Ejercicio 3: Iris":
        st.info("**Objetivo:** Implementar flujo completo de preprocesamiento y visualización.")
    
    st.markdown("---")
    st.markdown("**Desarrollado con:**")
    st.markdown("- Python 🐍")
    st.markdown("- Streamlit 🎈")
    st.markdown("- Scikit-learn 🤖")
    st.markdown("- Pandas 🐼")

# Contenido principal
if ejercicio_seleccionado == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Procesamiento de Datasets")
    
    st.markdown("""
    ## 🎯 Objetivo General
    
    Aplicar las etapas del procesamiento de datos sobre diferentes conjuntos de datos reales 
    usando Python y bibliotecas como **pandas** y **scikit-learn**.
    
    ## 📝 Etapas del Procesamiento
    
    En cada ejercicio se implementan las siguientes etapas:
    
    1. **Carga del dataset** 📥
    2. **Exploración inicial** 🔍 (info, describe, nulls, tipos de datos)
    3. **Limpieza de datos** 🧹 (valores nulos, duplicados, outliers)
    4. **Codificación de variables categóricas** 🔢
    5. **Normalización o estandarización** ⚖️
    6. **División en conjuntos de entrenamiento y prueba** ✂️
    
    ## 🎓 Ejercicios Disponibles
    
    ### 🚢 Ejercicio 1: Dataset Titanic
    Análisis y preparación de datos para predecir la supervivencia de los pasajeros del Titanic.
    
    ### 📚 Ejercicio 2: Student Performance
    Procesamiento de datos de rendimiento estudiantil para predecir notas finales.
    
    ### 🌸 Ejercicio 3: Dataset Iris
    Implementación completa de preprocesamiento con visualizaciones del famoso dataset Iris.
    
    ---
    
    👈 **Selecciona un ejercicio del menú lateral para comenzar**
    """)
    
    # Mostrar imagen o información adicional
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ejercicios", "3", help="Tres ejercicios completos de procesamiento")
    
    with col2:
        st.metric("Datasets", "3", help="Titanic, Student Performance e Iris")
    
    with col3:
        st.metric("Etapas", "6", help="Seis etapas de procesamiento por ejercicio")

elif ejercicio_seleccionado == "🚢 Ejercicio 1: Titanic":
    mostrar_ejercicio1()

elif ejercicio_seleccionado == "📚 Ejercicio 2: Student Performance":
    mostrar_ejercicio2()

elif ejercicio_seleccionado == "🌸 Ejercicio 3: Iris":
    mostrar_ejercicio3()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Actividad Individual - Procesamiento de Datasets en Machine Learning</div>",
    unsafe_allow_html=True
)
