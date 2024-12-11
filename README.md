# Exploratory Data Analysis (EDA) - Predicción de precios de viviendas en Japón

## Descripción del Proyecto
Este proyecto tiene como objetivo realizar un análisis exploratorio de datos (EDA) para predecir los precios de venta de viviendas en Japón. Se utilizan varios algoritmos de machine learning para identificar el modelo más adecuado basado en su desempeño en diferentes métricas de evaluación.

## Estructura del Proyecto
- **Carga de datos**: Lectura y procesamiento de los datos desde archivos en formato Parquet.
- **Análisis exploratorio**: Visualizaciones y estadísticas descriptivas para entender las características de los datos.
- **Modelado predictivo**: Entrenamiento y evaluación de varios modelos, incluyendo ajuste de hiperparámetros.

## Tecnologías y Bibliotecas Utilizadas
- **Lenguaje**: Python 3.10.9
- **Bibliotecas**:
  - Exploración y visualización: `pandas`, `numpy`, `matplotlib`, `seaborn`
  - Modelos de machine learning: `scikit-learn`, `CatBoost`, `XGBoost`, `KNN`, `Random Forest`
  - Optimización de hiperparámetros: `optuna`

## Modelos Evaluados
- **CatBoostRegressor**
- **XGBoostRegressor**
- **RandomForestRegressor**
- **KNeighborsRegressor**

## Resultados y Conclusiones
1. **Mejores Modelos**:
   - **CatBoost** y **XGBoost** se destacaron como los modelos con mejor desempeño.
     - **Métrica R² (CV)** y **R² (Val)** superiores al 0.82.
     - CatBoost fue líder en validación cruzada, mientras que XGBoost tuvo más consistencia en validación.

2. **Modelos Menos Adecuados**:
   - **KNN** tuvo un desempeño bajo, con un R² de aproximadamente 0.31 y errores significativos (MAE y RMSE).

3. **Impacto del Ajuste de Hiperparámetros**:
   - No se observó una mejora significativa tras el tuning, posiblemente por:
     - Rangos de búsqueda poco amplios.
     - Modelos ya en un punto óptimo.
     - Limitaciones en el tamaño de la muestra de datos.

4. **Recomendaciones**:
   - Utilizar **CatBoost** o **XGBoost** como modelos finales.
   - Revisar los rangos de hiperparámetros para mayor refinamiento.
   - Incrementar el tamaño del conjunto de datos para mejorar el ajuste.

**Decisión Final**: Entre **CatBoost** y **XGBoost**, la elección depende del balance entre velocidad y rendimiento. CatBoost es más preciso en validación cruzada, mientras que XGBoost ofrece más estabilidad en validación.

## Uso del Proyecto
1. Clona este repositorio:
   ```bash
   git clone <URL del repositorio>
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el notebook `EDA.ipynb` para replicar el análisis.

## Estructura del Repositorio
```
/
├── EDA.ipynb               # Notebook principal del proyecto
├── data/                   # Carpeta con los datos (crudos y procesados)
├── models/                 # Modelos entrenados
├── functions/              # Funciones personalizadas
└── requirements.txt        # Dependencias del proyecto
```

## Contacto
Para dudas o sugerencias, por favor contacta a Rodrigo.

