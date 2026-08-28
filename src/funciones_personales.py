from datetime import date
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
#=================================================================
def mostrar_nan(df, columna=None, mostrar=False, num_filas=10):
    """
    Imprime la cantidad de valores Nan por columna del DataFrame
    Muestra las filas completas que contienen valores NaN
    
    Parámetros:
    df: DataFrame a analizar
    columna: nombre de columna específica (opcional)
    num_filas: cuántas filas mostrar (por defecto 10)
    """
    name = str(df)
    lista= df.columns

    #Para una sola columna:
    if columna:  
        filas_nan = df[df[columna].isna()] #crea DF de puros valores NaN
        if mostrar:
            print(f"\nFilas con NaN en '{columna}': {len(filas_nan)} encontradas") # imprime nombre de la columna y cantidad de NaN
        if not filas_nan.empty: # Si el DF filas_nan tiene algo adentro
            if mostrar:
                print("\nPrimeras", min(num_filas, len(filas_nan)), "filas completas conteniendo NaN para: ", "'",columna,"'")
                print(filas_nan.head(num_filas))
                print('-'*50) #separador visual
    else:
        filas_nan =df[df.isna().any(axis=1)]
        if mostrar:
            # 1. Mostrar total de filas con NaN
            print(f"\nTotal de filas con NaN: {len(filas_nan)}")
    
            # 2. Mostrar NaN por cada columna del DataFrame ORIGINAL
            print(f"\nCantidad de NaN por columna:")
            for column in df.columns:  # df.columns, no filas_nan.columns
                nan_count = df[column].isna().sum()  # Contar NaN en cada columna
                print(f"  {column}: {nan_count}")
    
            # 3. Mostrar las primeras filas con NaN
            if not filas_nan.empty:
                print(f"\nPrimeras {min(num_filas, len(filas_nan))} filas con NaN:")
                print(filas_nan.head(num_filas))
                print('-'*50)
    return filas_nan

#========================================
def evaluar_modelo(
    modelo,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo,
    parametros=None,
    cat_features=None
):
    """
    Entrena un modelo y devuelve sus métricas de desempeño.

    Parámetros
    ----------
    modelo : estimador de sklearn o compatible
    X_train, y_train : datos de entrenamiento
    X_valid, y_valid : datos de validación
    nombre_modelo : str
    parametros : dict, opcional
        Diccionario con los hiperparámetros utilizados.

    Retorna
    -------
    DataFrame con una fila de resultados.
    """

    # ==========================
    # Entrenamiento
    # ==========================
    inicio_train = time.perf_counter()

    if cat_features is not None:
        modelo.fit(
            X_train,
            y_train,
            cat_features=cat_features
        )
    else:
        modelo.fit(
            X_train,
            y_train
        )

    tiempo_train = time.perf_counter() - inicio_train


    # ==========================
    # Predicción
    # ==========================
    inicio_pred = time.perf_counter()

    predicciones = modelo.predict(X_valid)

    tiempo_pred = time.perf_counter() - inicio_pred


    # ==========================
    # RMSE
    # ==========================
    rmse = mean_squared_error(y_valid, predicciones, squared=False)
    
    
    # ==========================
    # Formato de hiperparámetros
    # ==========================
    
    if parametros is None:
        parametros_txt = "Default"
    else:
        parametros_txt = ", ".join(
            f"{k}={v}" for k, v in parametros.items()
        )

    
    # ==========================
    # Resultados
    # ==========================
    resultados = pd.DataFrame({
        'Modelo': [nombre_modelo],
        'Parámetros': [parametros_txt],
        'RMSE': [rmse],
        'Tiempo entrenamiento (s)': [tiempo_train],
        'Tiempo predicción (s)': [tiempo_pred]
    })

    return resultados, modelo
#====================================================
def probar_hiperparametros(
    modelo_base,
    lista_parametros,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo,
    column_name,
    lista_features
):

    resultados = []

    mejor_modelo = None
    mejor_rmse = float("inf")

    for parametros_features in lista_features:

        max_lag = parametros_features['max_lag']
        rolling_size = parametros_features['rolling_mean_size']

        contexto_size = max(max_lag, rolling_size)

        # ==========================================
        # Crear series con el target
        # ==========================================

        train_data = pd.DataFrame(
            {column_name: y_train},
            index=X_train.index
        )

        valid_data = pd.DataFrame(
            {column_name: y_valid},
            index=X_valid.index
        )

        # ==========================================
        # Contexto del train para validación
        # ==========================================

        contexto = train_data.iloc[-contexto_size:].copy()

        valid_con_contexto = pd.concat([
            contexto,
            valid_data
        ])
        # ==========================================
        # Crear features
        # ==========================================


        valid_features = make_features(
            valid_con_contexto,
            column_name,
            max_lag,
            rolling_size
        )
        valid_features = valid_features.iloc[contexto_size:]


        
        train_features = make_features(
            train_data,
            column_name,
            max_lag,
            rolling_size
        )

        
        X_train_features = train_features.drop(columns=[column_name])
        X_valid_features = valid_features.drop(columns=[column_name])

        # Eliminar NaN generados por lag y rolling
        X_train_features = X_train_features.dropna()
        X_valid_features = X_valid_features.dropna()

        # Alinear y
        y_train_features = y_train.loc[X_train_features.index]
        y_valid_features = y_valid.loc[X_valid_features.index]

        for parametros in lista_parametros:

            modelo = modelo_base(**parametros)

            resultado, modelo_entrenado = evaluar_modelo(
                modelo=modelo,
                X_train=X_train_features,
                y_train=y_train_features,
                X_valid=X_valid_features,
                y_valid=y_valid_features,
                nombre_modelo=nombre_modelo,
                parametros=parametros
            )

            # Agregar parámetros de features
            resultado['max_lag'] = parametros_features['max_lag']
            resultado['rolling_mean_size'] = parametros_features['rolling_mean_size']

            resultados.append(resultado)

            if resultado.loc[0, 'RMSE'] < mejor_rmse:

                mejor_rmse = resultado.loc[0, 'RMSE']
                mejor_modelo = modelo_entrenado

    resultados = (
        pd.concat(resultados, ignore_index=True)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    return resultados, mejor_modelo

#===============================
def descomponer_plots(df, date_inicio=None, date_final=None):
    
    # Asegurarnos de que el índice sea datetime
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Ordenar por fecha
    df = df.sort_index()
    
    # Seleccionar el período solicitado
    if date_inicio is not None:
        df = df.loc[date_inicio:]
        
    if date_final is not None:
        df = df.loc[:date_final]
    
    # Descomponer la serie
    decomposed = seasonal_decompose(df)
    
    # Graficar
    plt.figure(figsize=(6, 8))
    
    plt.subplot(311)
    decomposed.trend.plot(ax=plt.gca())
    plt.title('Trend')
    
    plt.subplot(312)
    decomposed.seasonal.plot(ax=plt.gca())
    plt.title('Seasonality')
    
    plt.subplot(313)
    decomposed.resid.plot(ax=plt.gca())
    plt.title('Residuals')
    
    plt.tight_layout()
    plt.show()

#===============================================

def make_features(data,column_name, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data[column_name].shift(lag)

    data['rolling_mean'] = data[column_name].shift().rolling(rolling_mean_size).mean()

    return data