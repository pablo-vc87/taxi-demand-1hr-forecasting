# Pronóstico de Demanda de Taxis a 1 Hora

> English version: [README.md](README.md)

Un proyecto de pronóstico de series de tiempo que predice el **número de pedidos de taxi para la próxima hora** en aeropuertos, a partir de la demanda histórica registrada cada 10 minutos.

El proyecto remuestrea los registros de alta frecuencia a una serie horaria, analiza la tendencia y la estacionalidad, y construye variables de calendario, rezagos y media móvil con una división cronológica estricta en entrenamiento, validación y prueba. El modelo seleccionado, **Regresión Lineal con 72 rezagos y una media móvil de 4 horas**, obtuvo un **RMSE de 30.72 en validación** y un **RMSE final de 42.48 en el conjunto de prueba independiente**, **28% menor que un pronóstico ingenuo** que repite la hora anterior (RMSE 58.86) y dentro del requisito de **RMSE ≤ 48**.

El mismo flujo de trabajo (agregar registros de alta frecuencia, construir variables con rezagos y validar sin mirar al futuro) se aplica a series de tiempo de sensores y datos ambientales, como mediciones meteorológicas o de calidad del aire registradas cada pocos segundos.

---

# Contexto de Negocio

Sweet Lift Taxi ha recopilado datos históricos de pedidos de taxi en aeropuertos. Para atraer más conductores en las horas pico, la empresa necesita predecir cuántos pedidos llegarán en la **próxima hora**. Un pronóstico confiable ayuda a:

* Anticipar los picos de demanda y programar conductores con tiempo.
* Reducir los tiempos de espera de los pasajeros en las horas de mayor demanda.
* Evitar conductores sin trabajo en los periodos de baja demanda.

El modelo debe predecir un objetivo:

| Objetivo | Descripción |
| --- | --- |
| `num_orders` | Número de pedidos de taxi durante la próxima hora |

---

# Conceptos de Series de Tiempo

Pronosticar la próxima hora depende de cómo se comportó la demanda en las horas anteriores y de la hora del día. El proyecto usa los siguientes conceptos:

| Término | Significado |
| --- | --- |
| Remuestreo | Agregar los registros de 10 minutos en totales por hora |
| Tendencia | Dirección de largo plazo de la serie (creciente durante el periodo) |
| Estacionalidad | Patrón que se repite en un intervalo fijo (aquí, cada día) |
| Residuos | Lo que queda después de quitar la tendencia y la estacionalidad |
| Rezago (`lag_n`) | Número de pedidos *n* horas antes de la hora actual |
| Media móvil | Promedio de las horas anteriores, desplazado para que nunca incluya el valor actual |
| División cronológica | Entrenamiento, validación y prueba en orden temporal, sin mezclar |

## Nombres de las Variables

```text
Calendario:    year, month, day, dayofweek, hour
Rezagos:       lag_1, lag_2, ..., lag_72      (pedidos de 1 a 72 horas antes)
Media móvil:   rolling_mean                   (media de las k horas anteriores)
```

---

# Métrica de Evaluación

El proyecto usa la **Raíz del Error Cuadrático Medio (RMSE)**, que se expresa en la misma unidad que el objetivo (pedidos por hora) y penaliza más los errores grandes que los pequeños. Esto importa aquí porque fallar un pico de demanda cuesta más que una desviación pequeña en una hora tranquila.

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(y_i - \hat{y}_i\right)^2}
$$

El requisito del proyecto era **RMSE ≤ 48** en el conjunto de prueba.

---

# Dataset

Los datos son pedidos históricos de taxi registrados cada **10 minutos** e indexados por fecha y hora (`datetime`).

| Archivo | Filas (por hora) | Columnas | Descripción |
| --- | ---: | ---: | --- |
| `data/taxi.csv` | 4,416 | 1 | Número de pedidos (`num_orders`) por marca de tiempo |

* Periodo cubierto: **1 de marzo al 31 de agosto de 2018**.
* Después de remuestrear a una hora, la serie tiene **4,416 observaciones** y **ningún valor faltante**.

**Origen de los datos.** El dataset fue proporcionado por **TripleTen** como parte de su Data Science Bootcamp y se incluye en este repositorio con fines educativos y de reproducibilidad.

---

# Flujo del Proyecto

```text
Datos originales (registros cada 10 minutos)
            │
            ▼
1. Preparación de datos
   ├── Índice datetime y orden cronológico
   └── Remuestreo por hora (suma de pedidos)
            │
            ▼
2. Análisis de la serie de tiempo
   ├── Tendencia, estacionalidad y residuos del periodo completo
   └── Ventanas de dos semanas (marzo y mayo) para ver los patrones diarios
            │
            ▼
3. División cronológica: 80% entrenamiento / 10% validación / 10% prueba
            │
            ▼
4. Ingeniería de variables: calendario + rezagos + media móvil desplazada
            │
            ▼
5. Modelado
   ├── Regresión Lineal: 16 combinaciones de rezagos y ventana
   ├── Random Forest: hiperparámetros y combinaciones de rezagos y ventana
   ├── LightGBM: 9 configuraciones de hiperparámetros
   └── Reentrenar el mejor modelo con entrenamiento + validación y evaluar una vez en prueba
```

---

# 1. Preparación de Datos

* La columna `datetime` se convirtió en el índice y los registros se **ordenaron cronológicamente**.
* Los registros de 10 minutos se **remuestrearon a totales por hora**, porque el objetivo es pronosticar la próxima hora:

```python
df.sort_index(inplace=True)
df = df.resample('1h').sum()
```

* La serie resultante tiene **4,416 observaciones horarias** entre 2018-03-01 y 2018-08-31, sin valores faltantes.

---

# 2. Análisis de la Serie de Tiempo

La serie se descompuso en **tendencia**, **estacionalidad** y **residuos** con `seasonal_decompose` (statsmodels), primero para el periodo completo y luego para ventanas de dos semanas en marzo y mayo.

* **Tendencia:** la demanda **aumenta** a lo largo del periodo. Es relativamente estable en marzo y crece con más claridad a partir de mayo.
* **Estacionalidad:** hay un **ciclo diario claro**, con varios picos durante el día; el pico más fuerte está en las **últimas horas del día**.
* **Residuos:** muestran picos puntuales, pero sin una estructura marcada en comparación con la componente estacional.

**Implicación para el modelado.** Como la próxima hora depende de las horas más recientes y de la hora del día, las variables se enfocan en la **historia reciente (rezagos y media móvil)** y en las **variables de calendario**.

---

# 3. División Cronológica

Los datos se dividieron **sin mezclar**, de modo que el modelo nunca aprende de observaciones posteriores a las que predice.

```text
100% de los datos (en orden temporal)
│
├── 80% ENTRENAMIENTO ──→ entrenamiento de modelos
├── 10% VALIDACIÓN ─────→ selección de variables e hiperparámetros
└── 10% PRUEBA ─────────→ evaluación final (se usa una sola vez)
```

El conjunto de prueba no se tocó durante la selección de variables, el ajuste de hiperparámetros ni la selección del modelo.

---

# 4. Ingeniería de Variables

| Grupo | Variables | Propósito |
| --- | --- | --- |
| Calendario | `year`, `month`, `day`, `dayofweek`, `hour` | Capturar patrones diarios y semanales |
| Rezagos | `lag_1` … `lag_n` | Demanda en las *n* horas anteriores |
| Media móvil | `rolling_mean` (ventana *k*) | Nivel reciente de la demanda |

* La media móvil se calcula sobre la serie **desplazada**, así que solo usa valores pasados y no filtra el objetivo actual.
* Se eliminaron las filas con valores faltantes generadas por los primeros rezagos.
* En validación y prueba se usaron como contexto las **últimas observaciones del periodo anterior** para construir los primeros rezagos, sin usar ningún valor futuro.

Los valores probados fueron `max_lag ∈ {4, 8, 24, 72}` y `rolling_mean_size ∈ {4, 8, 24, 72}`. La mejor combinación fue **72 rezagos con una media móvil de 4 horas**.

---

# 5. Modelado

Se entrenaron tres modelos de regresión con el conjunto de entrenamiento y se compararon en el de validación:

| Modelo | Configuraciones probadas | Mejor RMSE en Validación |
| --- | --- | ---: |
| **Regresión Lineal** | 16 combinaciones de rezagos y ventana | **30.72** |
| Random Forest Regressor | 400 árboles con profundidad máxima 10, 15 y 20 (72 rezagos, ventana de 4 horas), luego 5 combinaciones de rezagos y ventana | 31.04 |
| LightGBM Regressor | 9 conjuntos de hiperparámetros (72 rezagos, ventana de 4 horas) | 48.03 |

* La **Regresión Lineal** obtuvo el menor error con **72 rezagos y una media móvil de 4 horas**.
* **Random Forest** (400 árboles, profundidad máxima 20) quedó cerca (31.04), pero no mejoró a la Regresión Lineal y necesitó mucho más tiempo de entrenamiento (unos 38 s contra 0.01 s). Al fijar sus hiperparámetros y cambiar los rezagos o la ventana, el resultado empeoró.
* **LightGBM** tuvo el resultado más débil (48.03), por encima del error de validación de los otros dos modelos.

## Resultado Final

La configuración seleccionada se reentrenó con **entrenamiento + validación (90% de los datos)** y se evaluó **una sola vez** en el conjunto de prueba. Las últimas 72 horas antes del periodo de prueba se usaron como contexto para construir los primeros rezagos de prueba.

```text
Modelo:               Regresión Lineal
Variables:            72 rezagos + media móvil de 4 horas + calendario
RMSE en validación:   30.72
RMSE en prueba:       42.48
Requisito:            RMSE ≤ 48   ✓
```

### Comparación con Modelos Base

Para medir cuánto aporta el modelo, se comparó en el mismo conjunto de prueba con dos pronósticos ingenuos que no requieren entrenamiento. Ambos usan las columnas de rezago ya construidas para la prueba, así que no usan información futura.

| Modelo | RMSE en prueba | Mejora del modelo final |
| --- | ---: | ---: |
| Base: hora anterior (`lag_1`) | 58.86 | 27.8% |
| Base: misma hora del día anterior (`lag_24`) | 57.19 | 25.7% |
| **Regresión Lineal (modelo final)** | **42.48** | — |

---

# Conclusiones

1. **El remuestreo por hora** convirtió los registros de 10 minutos en una serie adecuada para pronosticar a una hora.

2. **La serie tiene una tendencia creciente y una estacionalidad diaria clara**, lo que justificó usar variables de calendario, rezagos y una media móvil.

3. **La división cronológica evitó la fuga de información**: el conjunto de prueba se usó una sola vez, después de seleccionar el modelo.

4. **La Regresión Lineal fue el mejor modelo**, con un RMSE de validación de **30.72**, frente a Random Forest (**31.04**) y LightGBM (**48.03**). Un modelo simple con variables temporales bien diseñadas superó a modelos más complejos y se entrenó en una fracción del tiempo.

5. **En el conjunto de prueba independiente, el RMSE fue de 42.48**, 27.8% menor que el modelo base de la hora anterior (58.86) y 25.7% menor que el de la misma hora del día anterior (57.19), y por debajo del límite de 48. La diferencia con validación (30.72) es coherente con la tendencia creciente: el conjunto de prueba cubre del **13 al 31 de agosto**, cuando la demanda es más alta y más variable (media de **139.6** pedidos por hora contra **74.3** en entrenamiento, y desviación estándar de 58.6 contra 35.7), así que los errores medidos en pedidos por hora son mayores.

## Siguientes Pasos

* Validar con divisiones temporales sucesivas (`TimeSeriesSplit`, disponible en scikit-learn) para comprobar que el resultado es estable en distintos periodos.
* Investigar el bajo resultado de LightGBM: los modelos basados en árboles no pueden extrapolar más allá de los valores del objetivo vistos en el entrenamiento, lo que puede perjudicarlos en una serie con tendencia creciente.

---

# Estructura del Proyecto

```text
taxi-demand-1hr-forecasting/
│
├── data/
│   └── taxi.csv
│
├── notebooks/
│   └── taxi_demand_1hr_forecasting.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
└── README.md
```

---

# Instalación

Clona el repositorio:

```bash
git clone https://github.com/pablo-vc87/taxi-demand-1hr-forecasting.git

cd taxi-demand-1hr-forecasting
```

## Opción 1: Conda

```bash
conda env create -f environment.yml
conda activate taxi_demand_forecasting
jupyter notebook
```

## Opción 2: pip

Crea y activa un entorno virtual, y después:

```bash
pip install -r requirements.txt
jupyter notebook
```

Abre `notebooks/taxi_demand_1hr_forecasting.ipynb` y ejecuta todas las celdas. Los experimentos con Random Forest tardan unos minutos en una laptop.

---

# Tecnologías

* Python
* Pandas
* NumPy
* Matplotlib
* statsmodels
* scikit-learn
* LightGBM
* Jupyter Notebook

---

# Habilidades Demostradas

* Remuestreo de series de tiempo (de registros de alta frecuencia a serie horaria)
* Descomposición de tendencia y estacionalidad
* Ingeniería de variables con rezagos y ventanas móviles
* División cronológica en entrenamiento, validación y prueba
* Prevención de fuga de información en series de tiempo
* Comparación de modelos y prueba de hiperparámetros
* Regresión Lineal, Random Forest y LightGBM
* Evaluación con RMSE en un conjunto de prueba independiente

---

## Autor

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
