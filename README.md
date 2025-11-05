# 📊 Machine Learning - Procesamiento de Datasets

Proyecto de Streamlit para el procesamiento de datasets en Machine Learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 Descripción

Esta aplicación implementa las etapas del procesamiento de datos (carga, exploración, limpieza, codificación, normalización y división de datos) sobre diferentes conjuntos de datos reales usando Python y bibliotecas como pandas y scikit-learn.

## 📁 Estructura del Proyecto

```
MachineLearningNV/
│
├── app.py                          # Aplicación principal de Streamlit
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Documentación
├── .gitignore                      # Archivos a ignorar en Git
│
├── .streamlit/                     # Configuración de Streamlit
│   └── config.toml                 # Tema y configuración del servidor
│
├── ui/                             # Vistas de Streamlit
│   ├── __init__.py
│   ├── ejercicio1_view.py         # Vista del ejercicio Titanic
│   ├── ejercicio2_view.py         # Vista del ejercicio Student Performance
│   └── ejercicio3_view.py         # Vista del ejercicio Iris
│
└── ejercicios/                     # Lógica de procesamiento
    ├── __init__.py
    ├── ejercicio1/                 # Procesamiento Titanic
    │   ├── __init__.py
    │   ├── procesamiento.py
    │   └── titanic.csv
    ├── ejercicio2/                 # Procesamiento Student Performance
    │   ├── __init__.py
    │   ├── procesamiento.py
    │   └── student-mat.csv
    └── ejercicio3/                 # Procesamiento Iris
        ├── __init__.py
        └── procesamiento.py
```

## 🚀 Instalación Local

1. **Clonar o navegar al directorio del proyecto:**
   ```bash
   cd c:\Users\Usuario\SistemasInteligentes\MachineLearningNV
   ```

2. **Crear un entorno virtual (recomendado):**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Ejecución Local

Para ejecutar la aplicación de Streamlit:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## ☁️ Despliegue en Streamlit Cloud

### Paso 1: Preparar el Repositorio en GitHub

1. **Inicializar Git (si no está inicializado):**
   ```bash
   git init
   ```

2. **Agregar todos los archivos:**
   ```bash
   git add .
   ```

3. **Hacer el primer commit:**
   ```bash
   git commit -m "Initial commit - Machine Learning Dataset Processing"
   ```

4. **Crear un repositorio en GitHub:**
   - Ve a [GitHub](https://github.com) y crea un nuevo repositorio
   - Nombre sugerido: `machine-learning-datasets`
   - No inicialices con README (ya tienes uno)

5. **Conectar y subir el repositorio:**
   ```bash
   git remote add origin https://github.com/TU_USUARIO/machine-learning-datasets.git
   git branch -M main
   git push -u origin main
   ```

### Paso 2: Desplegar en Streamlit Cloud

1. **Ir a Streamlit Cloud:**
   - Visita [share.streamlit.io](https://share.streamlit.io)
   - Inicia sesión con tu cuenta de GitHub

2. **Crear Nueva App:**
   - Haz clic en "New app"
   - Selecciona tu repositorio: `TU_USUARIO/machine-learning-datasets`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL (personalizada): `ml-datasets-processing` (o el nombre que prefieras)

3. **Desplegar:**
   - Haz clic en "Deploy!"
   - Espera a que se instalen las dependencias (2-3 minutos)
   - ¡Tu app estará en línea!

### Paso 3: Actualizar la App

Cuando hagas cambios en el código:

```bash
git add .
git commit -m "Descripción de los cambios"
git push
```

Streamlit Cloud detectará los cambios automáticamente y redesplegará la aplicación.

## 📚 Ejercicios Implementados

### 🚢 Ejercicio 1: Dataset Titanic
- **Objetivo:** Preparar datos para predecir la supervivencia de pasajeros
- **Dataset:** titanic.csv (891 registros)
- **Técnicas:** 
  - Limpieza de datos (valores nulos, duplicados)
  - Label Encoding (Sex, Embarked)
  - Estandarización (Age, Fare)
  - División 70/30

### 📖 Ejercicio 2: Student Performance
- **Objetivo:** Predecir la nota final (G3) de estudiantes
- **Dataset:** student-mat.csv (395 registros)
- **Técnicas:** 
  - One-Hot Encoding (variables categóricas)
  - Normalización MinMaxScaler
  - Análisis de correlación (G1, G2, G3)
  - División 80/20

### 🌸 Ejercicio 3: Dataset Iris
- **Objetivo:** Flujo completo de preprocesamiento con visualización
- **Dataset:** load_iris() de scikit-learn (150 registros)
- **Técnicas:** 
  - Estandarización StandardScaler
  - Visualización por clase
  - División 70/30

## 🔧 Etapas del Procesamiento

Cada ejercicio implementa las siguientes etapas:

1. **Carga del dataset** 📥
2. **Exploración inicial** 🔍
3. **Limpieza de datos** 🧹
4. **Codificación de variables categóricas** 🔢
5. **Normalización/Estandarización** ⚖️
6. **División train/test** ✂️

## 🛠️ Tecnologías Utilizadas

- **Python 3.10+**
- **Streamlit 1.28.0** - Framework web interactivo
- **Pandas 2.1.0** - Manipulación de datos
- **Scikit-learn 1.3.0** - Machine Learning
- **Matplotlib 3.7.2** - Visualización
- **Seaborn 0.12.2** - Visualización estadística
- **NumPy 1.24.3** - Cálculos numéricos

## 📊 Características de la Aplicación

- ✨ Interfaz interactiva con Streamlit
- 📈 Visualizaciones dinámicas con Matplotlib y Seaborn
- 📊 Tablas interactivas con Pandas
- 💾 Descarga de datasets procesados en CSV
- 🎨 Diseño responsive y profesional
- 🔄 Procesamiento en tiempo real
- 📱 Compatible con móviles y tablets

## 🐛 Solución de Problemas

### Error: "No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### Error: "This app has gone over its resource limits"
- Streamlit Cloud tiene límites de recursos
- Considera optimizar el código o usar Streamlit Cloud Plus

### La app no se actualiza después de hacer push
- Ve a Streamlit Cloud → Tu App → Menú (⋮) → "Reboot app"

## 👨‍💻 Autor

Actividad Individual - Sistemas Inteligentes

## 📄 Licencia

Este proyecto es de uso educativo.

## 🔗 Enlaces Útiles

- [Documentación de Streamlit](https://docs.streamlit.io)
- [Streamlit Cloud](https://share.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Pandas Documentation](https://pandas.pydata.org)

---

**Nota:** Asegúrate de que todos los archivos CSV estén en el repositorio antes de desplegar en Streamlit Cloud.
