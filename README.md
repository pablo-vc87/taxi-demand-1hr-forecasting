# Taxi Demand 1-Hour Forecasting

> Versión en español: [LEEME.md](LEEME.md)

A time series forecasting project that predicts the **number of taxi orders for the next hour** at airports, using historical demand recorded every 10 minutes.

The project resamples high-frequency records into an hourly series, analyzes trend and seasonality, and builds calendar, lag, and rolling-mean features with a strictly chronological train/validation/test split. The selected model, **Linear Regression with 72 lag features and a 4-hour rolling mean**, achieved a **validation RMSE of 30.72** and a **final RMSE of 42.48 on the independent test set**, **28% lower than a naive forecast** that repeats the previous hour (RMSE 58.86), and within the requirement of **RMSE ≤ 48**.

The same workflow (aggregating high-frequency records, building lagged features, and validating without looking into the future) applies to sensor and environmental time series, such as weather or air-quality measurements recorded every few seconds.

---

# Business Context

Sweet Lift Taxi has collected historical data on taxi orders at airports. To attract more drivers during peak hours, the company needs to predict how many orders will arrive in the **next hour**. A reliable forecast helps to:

* Anticipate demand peaks and schedule drivers in advance.
* Reduce passenger waiting times during busy hours.
* Avoid having idle drivers during low-demand periods.

The model must predict one target:

| Target | Description |
| --- | --- |
| `num_orders` | Number of taxi orders during the next hour |

---

# Time Series Concepts

Forecasting the next hour depends on how demand behaved in the previous hours and on the time of day. The project uses the following concepts:

| Term | Meaning |
| --- | --- |
| Resampling | Aggregating the 10-minute records into hourly totals |
| Trend | Long-term direction of the series (increasing during the period) |
| Seasonality | Pattern that repeats at a fixed interval (here, every day) |
| Residuals | What remains after removing trend and seasonality |
| Lag feature (`lag_n`) | Number of orders *n* hours before the current hour |
| Rolling mean | Average of the previous hours, shifted so it never includes the current value |
| Chronological split | Train, validation, and test sets kept in time order, never shuffled |

## Feature Naming Convention

```text
Calendar:      year, month, day, dayofweek, hour
Lags:          lag_1, lag_2, ..., lag_72      (orders 1 to 72 hours before)
Rolling mean:  rolling_mean                   (mean of the previous k hours)
```

---

# Evaluation Metric

The project uses the **Root Mean Squared Error (RMSE)**, which is expressed in the same unit as the target (orders per hour) and penalizes large errors more than small ones. This matters here because missing a demand peak is more costly than a small deviation in a quiet hour.

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(y_i - \hat{y}_i\right)^2}
$$

The project requirement was **RMSE ≤ 48** on the test set.

---

# Dataset

The data are historical taxi orders recorded every **10 minutes** and indexed by date and time (`datetime`).

| File | Rows (hourly) | Columns | Description |
| --- | ---: | ---: | --- |
| `data/taxi.csv` | 4,416 | 1 | Number of orders (`num_orders`) per timestamp |

* Period covered: **March 1 – August 31, 2018**.
* After resampling to one hour, the series has **4,416 observations** and **no missing values**.

**Data source.** The dataset was provided by **TripleTen** as part of its Data Science Bootcamp and is included in this repository for educational and reproducibility purposes.

---

# Project Workflow

```text
Raw data (10-minute records)
            │
            ▼
1. Data preparation
   ├── Datetime index and chronological sorting
   └── Hourly resampling (sum of orders)
            │
            ▼
2. Time series analysis
   ├── Full-period trend, seasonality, and residuals
   └── Two-week windows (March and May) to inspect daily patterns
            │
            ▼
3. Chronological split: 80% train / 10% validation / 10% test
            │
            ▼
4. Feature engineering: calendar + lags + shifted rolling mean
            │
            ▼
5. Modeling
   ├── Linear Regression: 16 lag / rolling-window combinations
   ├── Random Forest: hyperparameters and lag / window combinations
   ├── LightGBM: 9 hyperparameter configurations
   └── Retrain best model on train + validation and evaluate once on test
```

---

# 1. Data Preparation

* The `datetime` column was parsed as the index and the records were **sorted chronologically**.
* The 10-minute records were **resampled to hourly totals**, because the goal is to forecast the next hour:

```python
df.sort_index(inplace=True)
df = df.resample('1h').sum()
```

* The resulting series has **4,416 hourly observations** between 2018-03-01 and 2018-08-31, with no missing values.

---

# 2. Time Series Analysis

The series was decomposed into **trend**, **seasonality**, and **residuals** with `seasonal_decompose` (statsmodels), first for the full period and then for two-week windows in March and May.

* **Trend:** demand **increases** over the period. It is relatively stable in March and grows more clearly from May onward.
* **Seasonality:** there is a **clear daily cycle**, with several peaks during the day; the strongest peak is in the **last hours of the day**.
* **Residuals:** show occasional spikes but no strong structure compared with the seasonal component.

**Implication for modeling.** Because the next hour depends on the most recent hours and on the time of day, the features focus on **recent history (lags and rolling mean)** and **calendar variables**.

---

# 3. Chronological Split

The data was split **without shuffling**, so the model never learns from observations that come after the ones it predicts.

```text
100% of the data (in time order)
│
├── 80% TRAIN ────────→ model training
├── 10% VALIDATION ───→ feature and hyperparameter selection
└── 10% TEST ─────────→ final evaluation (used only once)
```

The test set stayed untouched during feature selection, hyperparameter tuning, and model selection.

---

# 4. Feature Engineering

| Group | Features | Purpose |
| --- | --- | --- |
| Calendar | `year`, `month`, `day`, `dayofweek`, `hour` | Capture daily and weekly patterns |
| Lags | `lag_1` … `lag_n` | Demand in the previous *n* hours |
| Rolling mean | `rolling_mean` (window *k*) | Recent demand level |

* The rolling mean is calculated on the **shifted** series, so it only uses past values and does not leak the current target.
* Rows with missing values created by the first lags were removed.
* For validation and test, the **last observations of the previous period** were used as context to build the first lags, without using any future values.

The tested values were `max_lag ∈ {4, 8, 24, 72}` and `rolling_mean_size ∈ {4, 8, 24, 72}`. The best combination was **72 lags with a 4-hour rolling mean**.

---

# 5. Modeling

Three regression models were trained on the training set and compared on the validation set:

| Model | Configurations tested | Best Validation RMSE |
| --- | --- | ---: |
| **Linear Regression** | 16 lag / rolling-window combinations | **30.72** |
| Random Forest Regressor | 400 trees with max depth 10, 15, and 20 (72 lags, 4-hour window), then 5 lag / window combinations | 31.04 |
| LightGBM Regressor | 9 hyperparameter sets (72 lags, 4-hour window) | 48.03 |

* **Linear Regression** obtained the lowest error with **72 lags and a 4-hour rolling mean**.
* **Random Forest** (400 trees, max depth 20) was close (31.04), but did not improve on Linear Regression and needed much more training time (about 38 s vs 0.01 s). Fixing its hyperparameters and changing the lags or window size made the result worse.
* **LightGBM** had the weakest result (48.03), above the validation error of the other two models.

## Final Result

The selected configuration was retrained on **train + validation (90% of the data)** and evaluated **once** on the test set. The last 72 hours before the test period were used as context to build the first test lags.

```text
Model:                Linear Regression
Features:             72 lags + 4-hour rolling mean + calendar
Validation RMSE:      30.72
Test RMSE:            42.48
Requirement:          RMSE ≤ 48   ✓
```

### Comparison with Baselines

To measure what the model adds, it was compared on the same test set with two naive forecasts that need no training. Both use the lag columns already built for the test set, so they do not use future information.

| Model | Test RMSE | Improvement of the final model |
| --- | ---: | ---: |
| Baseline: previous hour (`lag_1`) | 58.86 | 27.8% |
| Baseline: same hour of the previous day (`lag_24`) | 57.19 | 25.7% |
| **Linear Regression (final model)** | **42.48** | — |

---

# Conclusions

1. **Hourly resampling** turned the 10-minute records into a series suited to one-hour forecasting.

2. **The series has an increasing trend and a clear daily seasonality**, which justified using calendar variables, lags, and a rolling mean.

3. **A chronological split prevented information leakage**: the test set was used only once, after selecting the model.

4. **Linear Regression was the best model**, with a validation RMSE of **30.72**, compared with Random Forest (**31.04**) and LightGBM (**48.03**). A simple model with well-designed temporal features outperformed more complex ones and trained in a fraction of the time.

5. **On the independent test set, the RMSE was 42.48**, 27.8% lower than the previous-hour baseline (58.86) and 25.7% lower than the same-hour-previous-day baseline (57.19), and below the required 48. The gap versus validation (30.72) is consistent with the increasing trend: the test set covers **August 13–31**, when demand is highest and most variable (mean of **139.6** orders per hour vs **74.3** in training, and standard deviation of 58.6 vs 35.7), so errors measured in orders per hour are larger.

## Next Steps

* Validate with rolling time-based splits (`TimeSeriesSplit`, available in scikit-learn) to check that the result is stable across different periods.
* Investigate the low LightGBM result: tree-based models cannot extrapolate beyond the target values seen in training, which may hurt them in a series with an increasing trend.

---

# Project Structure

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

# Installation

Clone the repository:

```bash
git clone https://github.com/pablo-vc87/taxi-demand-1hr-forecasting.git

cd taxi-demand-1hr-forecasting
```

## Option 1: Conda

```bash
conda env create -f environment.yml
conda activate taxi_demand_forecasting
jupyter notebook
```

## Option 2: pip

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/taxi_demand_1hr_forecasting.ipynb` and run all cells. The Random Forest experiments take a few minutes on a laptop.

---

# Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* statsmodels
* scikit-learn
* LightGBM
* Jupyter Notebook

---

# Skills Demonstrated

* Time series resampling (high-frequency records to hourly series)
* Trend and seasonality decomposition
* Lag and rolling-window feature engineering
* Chronological train / validation / test splitting
* Data leakage prevention in time series
* Model comparison and hyperparameter testing
* Linear Regression, Random Forest, and LightGBM
* RMSE evaluation on an independent test set

---

## Author

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
