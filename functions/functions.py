import sys
import os
# Añadir el directorio 'functions' al sys.path
sys.path.append(os.path.abspath(os.path.join('..', 'functions')))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import optuna
import pickle

import warnings
warnings.filterwarnings("ignore")

def process_and_save_csv_to_parquet(raw_folder="raw", processed_folder="processed", file_prefix="trade_prices", 
                                    output_file_name="merged_trade_prices.parquet", csv_count=47):
    """
    Procesa múltiples archivos CSV, los concatena en un DataFrame, y guarda el resultado en formato Parquet.

    Args:
        raw_folder (str): Subcarpeta dentro de "data" que contiene los archivos CSV crudos.
        processed_folder (str): Subcarpeta dentro de "data" donde se guardará el archivo procesado.
        file_prefix (str): Prefijo común de los archivos CSV.
        output_file_name (str): Nombre del archivo Parquet de salida.
        csv_count (int): Cantidad de archivos CSV a procesar.

    Returns:
        None
    """
    # Construir la ruta hacia la carpeta de datos crudos
    base_path = os.path.dirname(os.getcwd())
    raw_data_path = os.path.join(base_path, "data", raw_folder, file_prefix)

    # Obtener lista de archivos CSV
    csv_files = [os.path.join(raw_data_path, f"{i:02}.csv") for i in range(1, csv_count + 1)]

    # Leer y concatenar archivos CSV
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Eliminar la columna "No" si existe
    if 'No' in concatenated_df.columns:
        concatenated_df = concatenated_df.drop('No', axis=1)

    # Mostrar información básica del DataFrame
    print(f"Shape of the concatenated DataFrame: {concatenated_df.shape}")
    print(concatenated_df.head())

    # Construir la ruta para guardar el archivo procesado
    processed_data_path = os.path.join(base_path, "data", processed_folder)
    os.makedirs(processed_data_path, exist_ok=True)

    # Guardar el DataFrame como archivo Parquet
    output_file_path = os.path.join(processed_data_path, output_file_name)
    concatenated_df.to_parquet(output_file_path, index=False)

    print(f"Archivo guardado exitosamente en: {output_file_path}")
    
    
def load_parquet_to_dataframe(folder_name="data", processed_folder="processed", file_name="merged_trade_prices.parquet"):
    """
    Carga un archivo Parquet y lo devuelve como un DataFrame de pandas.

    Args:
        folder_name (str): Nombre de la carpeta principal donde se encuentra la carpeta 'processed'.
        processed_folder (str): Nombre de la subcarpeta que contiene el archivo procesado.
        file_name (str): Nombre del archivo Parquet a cargar.

    Returns:
        pd.DataFrame: El DataFrame cargado desde el archivo Parquet.
    """
    # Construir la ruta completa del archivo Parquet
    base_path = os.path.dirname(os.getcwd())
    processed_data_path = os.path.join(base_path, folder_name, processed_folder)
    parquet_file_path = os.path.join(processed_data_path, file_name)

    # Leer el archivo Parquet
    df = pd.read_parquet(parquet_file_path)

    # Mostrar información básica del DataFrame
    print(f"Shape of the DataFrame: {df.shape}")
    print(df.head())

    return df


def analyze_price_trends(df):
    """
    Crea un gráfico de las tendencias de precios a lo largo de los últimos 10 años.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos de bienes raíces.

    Returns:
        None
    """
    # Filtrar datos para los últimos 10 años y propiedades residenciales
    filtered_df = df[(df['Year'] >= (df['Year'].max() - 10)) & (df['Type'].str.contains("Residential", na=False))]
    
    # Agrupar por año y calcular precios medios y medianos
    trend_data = filtered_df.groupby('Year')['TradePrice'].agg(['mean', 'median']).reset_index()

    # Visualización: Tendencias de precios a lo largo del tiempo
    plt.figure(figsize=(10, 6))
    plt.plot(trend_data['Year'], trend_data['mean'], label='Precio promedio', marker='o', color='blue')
    plt.plot(trend_data['Year'], trend_data['median'], label='Precio mediano', marker='o', color='green')
    plt.title('Tendencias de precios de bienes raíces residenciales (últimos 10 años)')
    plt.xlabel('Año')
    plt.ylabel('Precio de comercio (TradePrice)')
    plt.legend()
    plt.grid()
    plt.show()
    
    
def compare_tokyo_vs_others(df):
    """
    Compara los precios de bienes raíces entre Tokyo y otras regiones.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos de bienes raíces.

    Returns:
        None
    """
    # Filtrar datos para Tokyo y otras regiones
    tokyo_df = df[df['Prefecture'] == 'Tokyo']
    others_df = df[df['Prefecture'] != 'Tokyo']
    
    # Calcular métricas para Tokyo y otras regiones
    tokyo_metrics = tokyo_df[['TradePrice', 'UnitPrice']].agg(['mean', 'median']).T
    others_metrics = others_df[['TradePrice', 'UnitPrice']].agg(['mean', 'median']).T
    print("Métricas para Tokyo:")
    print(tokyo_metrics)
    print("\nMétricas para otras regiones:")
    print(others_metrics)

    # Gráficos de comparación: Boxplots para precios de comercio
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=[tokyo_df['TradePrice'], others_df['TradePrice']], palette='Set2')
    plt.xticks([0, 1], ['Tokyo', 'Otras regiones'])
    plt.title('Distribución de precios de comercio entre Tokyo y otras regiones')
    plt.ylabel('Precio de comercio (TradePrice)')
    plt.show()

    # Gráficos de comparación: Barras para precios unitarios
    plt.figure(figsize=(12, 6))
    mean_prices = [tokyo_df['UnitPrice'].mean(), others_df['UnitPrice'].mean()]
    median_prices = [tokyo_df['UnitPrice'].median(), others_df['UnitPrice'].median()]
    x_labels = ['Tokyo', 'Otras regiones']
    
    bar_width = 0.35
    x = range(len(x_labels))

    plt.bar(x, mean_prices, width=bar_width, label='Precio promedio', color='blue')
    plt.bar([p + bar_width for p in x], median_prices, width=bar_width, label='Precio mediano', color='green')
    
    plt.xticks([p + bar_width / 2 for p in x], x_labels)
    plt.title('Comparación de precios unitarios entre Tokyo y otras regiones')
    plt.ylabel('Precio unitario (UnitPrice)')
    plt.legend()
    plt.show()
    
    
def compare_tokyo_vs_others_with_outlier_removal(df):
    """
    Compara los precios de bienes raíces entre Tokyo y otras regiones, excluyendo outliers.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos de bienes raíces.

    Returns:
        None
    """
    

    # Filtrar datos para Tokyo y otras regiones
    tokyo_df = df[df['Prefecture'] == 'Tokyo']
    others_df = df[df['Prefecture'] != 'Tokyo']

    # Definir función para eliminar outliers usando el rango intercuartílico (IQR)
    def remove_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Eliminar outliers para 'TradePrice' en cada subconjunto
    tokyo_df = remove_outliers(tokyo_df, 'TradePrice')
    others_df = remove_outliers(others_df, 'TradePrice')

    # Calcular métricas para Tokyo y otras regiones
    tokyo_metrics = tokyo_df[['TradePrice', 'UnitPrice']].agg(['mean', 'median']).T
    others_metrics = others_df[['TradePrice', 'UnitPrice']].agg(['mean', 'median']).T
    print("Métricas para Tokyo (sin outliers):")
    print(tokyo_metrics)
    print("\nMétricas para otras regiones (sin outliers):")
    print(others_metrics)

    # Gráficos de comparación: Boxplots para precios de comercio
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=[tokyo_df['TradePrice'], others_df['TradePrice']], palette='Set2')
    plt.xticks([0, 1], ['Tokyo', 'Otras regiones'])
    plt.title('Distribución de precios de comercio entre Tokyo y otras regiones (sin outliers)')
    plt.ylabel('Precio de comercio (TradePrice)')
    plt.show()

    # Eliminar outliers para 'TradePrice' en cada subconjunto
    tokyo_df = remove_outliers(tokyo_df, 'UnitPrice')
    others_df = remove_outliers(others_df, 'UnitPrice')

    # Gráficos de comparación: Barras para precios unitarios
    plt.figure(figsize=(12, 6))
    mean_prices = [tokyo_df['UnitPrice'].mean(), others_df['UnitPrice'].mean()]
    median_prices = [tokyo_df['UnitPrice'].median(), others_df['UnitPrice'].median()]
    x_labels = ['Tokyo', 'Otras regiones']
    
    bar_width = 0.35
    x = range(len(x_labels))

    plt.bar(x, mean_prices, width=bar_width, label='Precio promedio', color='blue')
    plt.bar([p + bar_width for p in x], median_prices, width=bar_width, label='Precio mediano', color='green')
    
    plt.xticks([p + bar_width / 2 for p in x], x_labels)
    plt.title('Comparación de precios unitarios entre Tokyo y otras regiones (sin outliers)')
    plt.ylabel('Precio unitario (UnitPrice)')
    plt.legend()
    plt.show()
    
    
def graficar_promedio_mediana_areas(df, columna_area="Area"):
    """
    Calcula y grafica el promedio y la mediana del área de las propiedades
    en Tokyo y otras regiones.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    columna_region : str
        Nombre de la columna que indica la región.
    columna_area : str
        Nombre de la columna que contiene las áreas de las propiedades.
    region_tokyo : str, opcional
        Nombre de la región que representa Tokyo (por defecto: "Tokyo").

    Retorna:
    --------
    None
    """
    # Separar datos por región
    tokyo_df = df[df['Prefecture'] == 'Tokyo']
    otras_df = df[df['Prefecture'] != 'Tokyo']

    # Calcular promedio y mediana
    promedios = {
        "Tokyo": tokyo_df[columna_area].mean(),
        "Otras regiones": otras_df[columna_area].mean()
    }
    medianas = {
        "Tokyo": tokyo_df[columna_area].median(),
        "Otras regiones": otras_df[columna_area].median()
    }

    # Crear un DataFrame para graficar
    resumen = pd.DataFrame({
        "Promedio": promedios,
        "Mediana": medianas
    })

    # Calcular métricas para Tokyo y otras regiones
    tokyo_metrics = tokyo_df[['Area']].agg(['mean', 'median']).T
    others_metrics = otras_df[['Area']].agg(['mean', 'median']).T
    print("Métricas para Tokyo:")
    print(tokyo_metrics)
    print("\nMétricas para otras regiones:")
    print(others_metrics)

    # Graficar los datos
    fig, ax = plt.subplots(figsize=(8, 6))
    resumen.plot(kind="bar", ax=ax, color=["skyblue", "salmon"])
    ax.set_title("Promedio y mediana de áreas por región", fontsize=14)
    ax.set_ylabel("Área (m²)", fontsize=12)
    ax.set_xlabel("Región", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Métrica", fontsize=10)
    plt.tight_layout()
    plt.show()
    
    
def remove_outliers(df, multiplier=1.5):
    """
    Elimina los outliers de todo el DataFrame usando el rango intercuartílico (IQR),
    ignorando columnas con valores faltantes.
    
    Parameters:
    df (DataFrame): DataFrame con los datos a procesar.
    multiplier (float): Factor para ajustar el rango de los outliers (default es 1.5).
    
    Returns:
    DataFrame: DataFrame sin los outliers.
    """
    # Crear una copia del DataFrame original
    df_cleaned = df.copy()
    
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        # Revisar si la columna tiene valores faltantes
        if df[column].isnull().any():
            print(f"Saltando columna '{column}' debido a valores faltantes.")
            continue
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Filtrar los datos dentro de los límites
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
        
        # Imprimir estadísticas
        print(f"Columna: {column}")
        print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
        print(f"Filas restantes: {df_cleaned.shape[0]}")
        print("-" * 50)
    
    return df_cleaned


def tomar_muestra_y_comparar(df, porcentaje_muestra):
    """
    Toma una muestra aleatoria del DataFrame completo, basada en un porcentaje del tamaño total,
    y realiza el Test de Kolmogorov-Smirnov (KS) para evaluar si la muestra es representativa 
    de la población original del DataFrame.

    Parámetros:
    df (DataFrame): DataFrame de entrada.
    porcentaje_muestra (float): Porcentaje del tamaño de la muestra (0-100).

    Retorna:
    ks_results (dict): Resultados KS por columna con el estadístico y el p-value.
    muestra (DataFrame): DataFrame de la muestra tomada.
    conclusion (str): Conclusión sobre si la muestra es representativa o no.
    """
    # Validar que el porcentaje esté en el rango válido
    if not (0 < porcentaje_muestra <= 100):
        raise ValueError("El porcentaje de muestra debe estar entre 0 y 100.")

    # Calcular el tamaño de la muestra basado en el porcentaje
    tamaño_muestra = int(len(df) * (porcentaje_muestra / 100))
    
    # Tomar una muestra aleatoria del DataFrame completo
    muestra = df.sample(n=tamaño_muestra, random_state=42)
    
    # Realizar el Test KS: Compara la muestra con la población original para cada columna numérica
    ks_results = {}
    
    # Iterar sobre las columnas numéricas del DataFrame
    for column in df.select_dtypes(include=[np.number]).columns:
        # Obtener la distribución empírica de la población original
        datos_poblacion = df[column].dropna()
        # Realizar el Test KS para la columna
        ks_statistic, p_value = stats.ks_2samp(muestra[column], datos_poblacion)
        
        # Guardar los resultados
        ks_results[column] = (ks_statistic, p_value)
    
    # Conclusión: Si el p-value de todas las columnas es mayor que 0.05, la muestra es representativa
    representativa = all(p_value > 0.05 for ks_statistic, p_value in ks_results.values())
    
    if representativa:
        conclusion = "La muestra es representativa de la población."
    else:
        conclusion = "La muestra no es representativa de la población."
    
    return ks_results, muestra, conclusion



def encode_categorical_columns(df, categorical_columns, encoding_type="label"):
    """
    Codifica columnas categóricas de un DataFrame utilizando OneHotEncoder o LabelEncoder.

    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas categóricas a codificar.
        categorical_columns (list): Lista con los nombres de las columnas categóricas.
        encoding_type (str): Tipo de codificación a aplicar. Puede ser "onehot" o "label".
                             - "onehot": Aplica OneHotEncoder.
                             - "label": Aplica LabelEncoder.

    Returns:
        pd.DataFrame: DataFrame con las columnas categóricas codificadas.
    """
    if encoding_type not in ["onehot", "label"]:
        raise ValueError("encoding_type debe ser 'onehot' o 'label'.")
    
    df_encoded = df.copy()

    if encoding_type == "onehot":
        # Aplicar OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_data = onehot_encoder.fit_transform(df_encoded[categorical_columns])
        
        # Crear un DataFrame con los nombres de las columnas codificadas
        encoded_columns = onehot_encoder.get_feature_names_out(categorical_columns)
        onehot_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df_encoded.index)
        
        # Concatenar con el DataFrame original (eliminando las columnas originales)
        df_encoded = pd.concat([df_encoded.drop(columns=categorical_columns), onehot_df], axis=1)

    elif encoding_type == "label":
        # Aplicar LabelEncoder
        label_encoder = LabelEncoder()
        for column in categorical_columns:
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))

    return df_encoded

def mostrar_importancia_variables(modelo, X):
    """
    Esta función obtiene la importancia de las características de un modelo entrenado
    y las visualiza en un gráfico de barras horizontales.

    Parámetros:
    modelo (sklearn.ensemble.RandomForestRegressor): El modelo entrenado (Random Forest).
    X (pd.DataFrame): El conjunto de datos con las características.

    Retorna:
    None
    """
    # Obtener la importancia de las características
    feature_importances = modelo.feature_importances_

    # Crear un DataFrame para visualizar la importancia de las características
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Mostrar los resultados
    print(importance_df)

    # Crear gráfico de barras horizontales
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importancia')
    plt.ylabel('Variables')
    plt.title('Importancia de las variables (Random Forest)')
    plt.gca().invert_yaxis()  
    plt.show()
    

def plot_correlation_matrix(df):
    """
    Grafica una matriz de correlación con una máscara que oculta la parte superior de la matriz.

    Parámetros:
    df (DataFrame): DataFrame que contiene los datos para calcular y graficar la matriz de correlación.

    Retorna:
    None
    """
    # Calcular la matriz de correlación
    correlations = df.corr()

    # Crear una máscara para la parte superior de la matriz
    mask = np.triu(np.ones_like(correlations, dtype=bool))

    # Configurar el tamaño de la figura
    plt.figure(figsize=(20, 10))

    # Graficar la matriz de correlación con la máscara aplicada
    sns.heatmap(
        correlations, 
        mask=mask, 
        annot=True, 
        cmap="YlGnBu", 
        fmt=".2f",  
        cbar_kws={"shrink": 0.8}  
    )

    # Mostrar el gráfico
    plt.title("Matriz de correlación")
    plt.show()
    
    
def save_df_to_parquet(df, folder="processed", output_file_name="output.parquet"):
    """
    Guarda un DataFrame en un archivo Parquet en la carpeta especificada.

    Args:
        df (pd.DataFrame): El DataFrame a guardar.
        folder (str): Subcarpeta dentro de "data" donde se guardará el archivo Parquet.
        output_file_name (str): Nombre del archivo Parquet de salida.

    Returns:
        None
    """
    # Obtener la ruta base
    base_path = os.path.dirname(os.getcwd())

    # Construir la ruta para guardar el archivo procesado
    processed_data_path = os.path.join(base_path, "data", folder)
    os.makedirs(processed_data_path, exist_ok=True)

    # Construir el path completo para el archivo Parquet
    output_file_path = os.path.join(processed_data_path, output_file_name)

    # Guardar el DataFrame como archivo Parquet
    df.to_parquet(output_file_path, index=False)

    print(f"Archivo guardado exitosamente en: {output_file_path}")
    
    
def plot_histograms(df):
    """
    Grafica histogramas con KDE para cada columna numérica en el DataFrame.

    Parámetros:
    df (DataFrame): DataFrame que contiene los datos a graficar.

    Retorna:
    None
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    num_columns = len(numerical_columns)

    # Definir el número de filas y columnas para la cuadrícula de subgráficos
    num_rows = (num_columns + 3) // 4 

    # Crear la figura y los ejes
    plt.figure(figsize=(15, 4 * num_rows))
    
    for i, column in enumerate(numerical_columns):
        ax = plt.subplot(num_rows, 4, i + 1)
        sns.histplot(data=df, x=column, kde=True, ax=ax)
        ax.set_title(f"Distribución de {column}")

    plt.tight_layout()
    plt.show()
    

def transformar_columnas_asimetricas(df, columnas):
    """
    Aplica una transformación logarítmica a las columnas especificadas con distribuciones asimétricas positivas.

    Parámetros:
    df (DataFrame): DataFrame que contiene los datos a transformar.
    columnas (list): Lista de nombres de las columnas a las que se aplicará la transformación.

    Retorna:
    DataFrame: El DataFrame con las columnas transformadas.
    """
    # Validar que las columnas existan en el DataFrame
    columnas_invalidas = [col for col in columnas if col not in df.columns]
    if columnas_invalidas:
        raise ValueError(f"Las siguientes columnas no están en el DataFrame: {columnas_invalidas}")

    # Definir el transformador logarítmico
    transformador_log = FunctionTransformer(np.log1p, validate=True)

    # Aplicar la transformación logarítmica a las columnas especificadas
    df[columnas] = transformador_log.transform(df[columnas])
    
    return df


def imputar_valores_nan(df, estrategia='mean'):
    """
    Imputa los valores NaN en el DataFrame utilizando SimpleImputer.

    Parámetros:
    df (DataFrame): El DataFrame con valores faltantes a imputar.
    estrategia (str): La estrategia de imputación. Puede ser 'mean', 'median', 'most_frequent' o 'constant'.

    Retorna:
    DataFrame: El DataFrame con los valores NaN imputados.
    """
    # Crear un imputador con la estrategia especificada
    imputador = SimpleImputer(strategy=estrategia)

    # Aplicar el imputador a los datos
    df_imputado = pd.DataFrame(
        imputador.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    return df_imputado


def dividir_dataset(df, target_col, test_size=0.2, val_size=0.1, random_state=42):
    """
    Divide el dataset en conjuntos de entrenamiento, prueba y validación.

    Args:
        df (DataFrame): Dataset completo.
        target_col (str): Columna objetivo.
        test_size (float): Tamaño del conjunto de prueba.
        val_size (float): Tamaño del conjunto de validación.
        random_state (int): Semilla para la reproducibilidad.

    Returns:
        tuple: Conjuntos X_train, X_test, X_val, y_train, y_test, y_val.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state
    )
    test_ratio = test_size / (test_size + val_size)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=test_ratio, random_state=random_state
    )
    return X_train, X_test, X_val, y_train, y_test, y_val


def evaluar_modelo(model, X_train, y_train, X_val, y_val, cv=5):
    """
    Evalúa un modelo utilizando validación cruzada y conjunto de validación.

    Args:
        model: Modelo a evaluar.
        X_train, y_train: Conjunto de entrenamiento.
        X_val, y_val: Conjunto de validación.
        cv (int): Número de pliegues para validación cruzada.

    Returns:
        dict: Métricas de evaluación.
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    
    # Ajustar modelo en el conjunto de entrenamiento completo
    model.fit(X_train, y_train)
    
    # Predecir en el conjunto de validación
    y_pred = model.predict(X_val)
    
    # Calcular métricas
    metrics = {
        'R2 (CV)': np.mean(cv_scores),
        'MAE': mean_absolute_error(y_val, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
        'R2 (Val)': r2_score(y_val, y_pred),
    }
    return metrics


def tune_model_random_forest(X_train, y_train, cv=3):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 20, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        }
        model = RandomForestRegressor(random_state=42, **params, n_jobs=-1)
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params


def tune_model_knn(X_train, y_train, cv=3):
    """
    Optimiza los hiperparámetros del modelo KNeighborsRegressor utilizando Optuna.

    Args:
        X_train (DataFrame): Conjunto de características de entrenamiento.
        y_train (Series): Etiquetas de entrenamiento.
        cv (int): Número de folds para validación cruzada.

    Returns:
        dict: Los mejores hiperparámetros encontrados por Optuna.
    """
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2),  # 1 para Manhattan, 2 para Euclidiana
        }
        model = KNeighborsRegressor(**params, n_jobs=-1)
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params


def tune_model_xgboost(X_train, y_train, cv=3):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        model = XGBRegressor(random_state=42, **params, n_jobs=-1)
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params


def tune_model_catboost(X_train, y_train, cv=3):
    """
    Optimiza los hiperparámetros del modelo CatBoost utilizando Optuna.

    Args:
        X_train (DataFrame): Conjunto de características de entrenamiento.
        y_train (Series): Etiquetas de entrenamiento.
        cv (int): Número de folds para validación cruzada.

    Returns:
        dict: Los mejores hiperparámetros encontrados por Optuna.
    """
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 200, step=50),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
        }
        model = CatBoostRegressor(**params, random_state=42, verbose=0,thread_count=-1)
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        return np.mean(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params


def comparar_modelos(modelos, X_train, y_train, X_val, y_val, tuned_params=None):
    """
    Compara el rendimiento de varios modelos antes y después de ajustar hiperparámetros.

    Args:
        modelos (dict): Diccionario de modelos iniciales.
        X_train, y_train: Conjunto de entrenamiento.
        X_val, y_val: Conjunto de validación.
        tuned_params (dict): Diccionario con parámetros ajustados para los modelos.

    Returns:
        DataFrame: Tabla comparativa de métricas.
    """
    resultados = []
    for nombre, modelo in modelos.items():
        # Evaluar modelo inicial
        metrics = evaluar_modelo(modelo, X_train, y_train, X_val, y_val)
        metrics['Modelo'] = nombre
        metrics['Hiperparámetros ajustados'] = False
        resultados.append(metrics)

        # Evaluar modelo con parámetros ajustados
        if tuned_params and nombre in tuned_params:
            if nombre == "CatBoost":
                # Crear una nueva instancia para CatBoost
                modelo_tuned = CatBoostRegressor(**tuned_params[nombre], random_state=42, verbose=0)
            else:
                # Para otros modelos, usar set_params
                modelo_tuned = modelo.set_params(**tuned_params[nombre])
            
            metrics = evaluar_modelo(modelo_tuned, X_train, y_train, X_val, y_val)
            metrics['Modelo'] = nombre
            metrics['Hiperparámetros ajustados'] = True
            resultados.append(metrics)

    return pd.DataFrame(resultados)

