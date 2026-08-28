# Taxi Demand 1-Hour Forecasting

A time series forecasting project focused on predicting the number of taxi orders for the next hour.

The project analyzes historical taxi demand data, explores temporal patterns, and compares different machine learning models using time-based features, lag variables, and rolling averages.

The final model, **Linear Regression with 72 lag features and a 4-hour rolling mean**, achieved a **final RMSE of 42.48 on the test set**, meeting the project requirement of **RMSE ≤ 48**.

---

# Project Overview

Sweet Lift Taxi has collected historical data on taxi orders at airports. To attract more drivers during peak hours, the company needs a system capable of predicting the number of taxi orders for the next hour.

The objective of this project was to build and evaluate machine learning models capable of forecasting hourly taxi demand.

The main performance requirement was:

```text
RMSE ≤ 48
```

The project included:

* Data preparation and hourly resampling.
* Exploratory analysis of trend and seasonality.
* Time-based feature engineering.
* Lag feature creation.
* Rolling mean features.
* Time-aware train, validation, and test splitting.
* Hyperparameter experimentation.
* Model comparison.
* Final retraining and independent test evaluation.

---

# Project Workflow

```text
Original Data (10-minute intervals)
                │
                ▼
     Datetime Index + Chronological Order
                │
                ▼
        Hourly Resampling
                │
                ▼
     Hourly Taxi Demand Time Series
                │
                ▼
      Exploratory Data Analysis
                │
        ┌───────┴────────┐
        ▼                ▼
      Trend        Seasonality
        │                │
        └───────┬────────┘
                ▼
       Time-Based Data Split
                │
                ▼
       Feature Engineering
                │
     ┌──────────┼───────────┐
     ▼          ▼           ▼
Calendar      Lags      Rolling Mean
Features
                │
                ▼
     Model + Hyperparameter Testing
                │
                ▼
       Validation Set Comparison
                │
                ▼
        Best Model Selection
                │
                ▼
 Retraining on Train + Validation
                │
                ▼
       Final Test Evaluation
                │
                ▼
           Conclusions
```

---

# Dataset

The dataset contains historical taxi order data.

**File:**

```text
data/taxi.csv
```

The original observations were recorded at **10-minute intervals**.

After preparing the datetime index and resampling the data to hourly intervals, the dataset contained:

| Feature         |        Value |
| --------------- | -----------: |
| Observations    |        4,416 |
| Columns         |            1 |
| Target variable | `num_orders` |
| Missing values  |            0 |
| Frequency       |       Hourly |
| Start date      |   2018-03-01 |
| End date        |   2018-08-31 |

Each observation after resampling represents the total number of taxi orders during one hour.

The hourly resampling was necessary because the objective of the project is to predict taxi demand for the **next hour**.

---

# Data Preparation

The original dataset was prepared using the following workflow:

```text
Original Data
     │
     ▼
Datetime Index
     │
     ▼
Chronological Sorting
     │
     ▼
Hourly Resampling
     │
     ▼
Hourly Taxi Demand Series
```

The data was sorted chronologically before resampling:

```python
df.sort_index(inplace=True)

df = df.resample('1H').sum()
```

After resampling, the dataset contained 4,416 hourly observations.

---

# Exploratory Analysis

The time series analysis showed two important patterns:

* A general increasing trend in taxi demand.
* A clear daily seasonality.

These patterns indicated that historical demand and temporal information could be useful for forecasting future taxi orders.

For this reason, the modeling process incorporated:

* Calendar features.
* Historical lag variables.
* Rolling mean features.

---

# Data Splitting

The data was divided while preserving chronological order.

Random shuffling was avoided because a forecasting model must only use information available before the prediction time.

The final 10% of the dataset was reserved exclusively for the final test evaluation.

```text
100% of the Data
│
├── 80% TRAIN ──────────────→ Model training
│
├── 10% VALIDATION ─────────→ Feature and model selection
│
└── 10% TEST ───────────────→ Final evaluation
```

The test set remained completely separate during:

* Feature selection.
* Lag configuration testing.
* Rolling mean testing.
* Hyperparameter tuning.
* Model selection.

Only after selecting the final configuration was the model evaluated on the test set.

---

# Feature Engineering

To predict the number of taxi orders for the next hour, several temporal and historical features were created.

## Calendar Features

The following features were extracted from the datetime index:

```text
year
month
day
dayofweek
hour
```

These variables allow the model to capture calendar and hourly patterns.

---

## Lag Features

Lag variables were created to provide information about previous taxi demand.

For example:

```text
lag_1
lag_2
lag_3
...
lag_n
```

The final selected configuration used:

```text
72 lags
```

This allowed the model to use information from the previous 72 hours.

---

## Rolling Mean

A shifted rolling mean was also included to represent recent demand levels.

The final selected configuration used:

```text
4-hour rolling mean
```

The target variable was shifted before calculating the rolling mean to prevent information leakage.

---

# Feature Engineering Pipeline

```text
Hourly Taxi Demand Series
            │
            ▼
     Calendar Features
(year, month, day, dayofweek, hour)
            │
            ▼
      Historical Lag Features
       (lag_1 ... lag_n)
            │
            ▼
     Shifted Rolling Mean
            │
            ▼
   Remove Initial Missing Values
            │
            ▼
Machine Learning Feature Matrix
```

---

# Models Evaluated

The following models were evaluated:

* Linear Regression
* Random Forest Regressor
* LightGBM Regressor

Different combinations of:

* Lag values.
* Rolling mean window sizes.
* Model hyperparameters.

were tested using the validation set.

---

# Model Performance

The best validation result for each model was:

| Model                 | RMSE Validation | Result            |
| --------------------- | --------------: | ----------------- |
| **Linear Regression** |       **30.72** | Best result       |
| Random Forest         |           31.04 | Did not improve   |
| LightGBM              |           48.03 | Lower performance |

The project requirement was:

```text
RMSE ≤ 48
```

Both Linear Regression and Random Forest achieved results below the required threshold during validation.

However, Linear Regression achieved the lowest validation RMSE.

---

# Linear Regression

Different combinations of lag values and rolling mean window sizes were evaluated.

The best configuration was:

```text
Model: Linear Regression

Maximum lag: 72 hours

Rolling mean window: 4 hours

Validation RMSE: 30.72
```

This result was significantly below the project requirement.

---

# Random Forest

Random Forest was evaluated using different hyperparameter configurations.

The best result was approximately:

```text
Validation RMSE: 31.04
```

Although the result was competitive, it did not improve upon Linear Regression.

Additionally, Random Forest required more training time.

Therefore, it was not selected as the final model.

---

# LightGBM

LightGBM was also evaluated using the engineered time-series features.

Its best validation result was:

```text
Validation RMSE: 48.03
```

This result was substantially worse than:

```text
Linear Regression: 30.72

Random Forest: 31.04
```

For this dataset and feature representation, LightGBM did not provide an improvement.

---

# Final Model Selection

The validation results showed that Linear Regression achieved the lowest error.

| Model                 | Validation RMSE |
| --------------------- | --------------: |
| **Linear Regression** |       **30.72** |
| Random Forest         |           31.04 |
| LightGBM              |           48.03 |

The selected configuration was:

```text
Model:
Linear Regression

Maximum lag:
72

Rolling mean:
4 hours
```

The model selection was performed exclusively using the training and validation datasets.

The test set was not used during model selection.

---

# Final Training

After selecting the best configuration, the final Linear Regression model was retrained using:

```text
TRAIN + VALIDATION

90% of the complete dataset
```

The final test set remained separate.

To create the first lag features for the test period, the last 72 observations from the training and validation period were used as historical context.

No future information from the test set was used when creating the initial prediction features.

---

# Final Evaluation Pipeline

```text
Final TRAIN + VALIDATION
            │
            ▼
 Context: Last 72 Observations
            │
            ▼
        TEST Dataset
            │
            ▼
Create 72 Lag Features
            │
            ▼
Create 4-Hour Rolling Mean
            │
            ▼
    Final Predictions
            │
            ▼
       RMSE Evaluation
```

---

# Final Results

The final model was evaluated once on the independent test set.

```text
Model:

Linear Regression

Features:

72 lag variables

4-hour rolling mean

Validation RMSE:

30.72

Final Test RMSE:

42.48

Project Requirement:

RMSE ≤ 48
```

## Result

```text
✓ PROJECT REQUIREMENT ACHIEVED
```

The final RMSE of **42.48** is below the maximum allowed value of **48**.

---

# Key Findings

The main findings of the project were:

1. The original data was successfully transformed from 10-minute intervals into an hourly time series.

2. The exploratory analysis revealed a general increasing trend and clear daily seasonality.

3. Time-based features, historical lags, and rolling means provided useful information for forecasting taxi demand.

4. Preserving chronological order during data splitting was essential to avoid future information leakage.

5. Linear Regression achieved the best validation performance with an RMSE of **30.72**.

6. The best feature configuration used **72 lag variables** and a **4-hour rolling mean**.

7. Random Forest achieved a competitive RMSE of **31.04**, but did not outperform Linear Regression and required more computational resources.

8. LightGBM produced the weakest result, with an RMSE of **48.03** during validation.

9. After retraining Linear Regression using the complete training and validation data, the final model achieved an RMSE of **42.48** on the independent test set.

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

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate taxi_demand_forecasting
```

Then start Jupyter Notebook:

```bash
jupyter notebook
```

---

## Option 2: pip

Create and activate a virtual environment.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

---

# Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* scikit-learn
* statsmodels
* LightGBM
* Jupyter Notebook

---

# Skills Demonstrated

This project demonstrates experience with:

* Time Series Forecasting
* Time Series Resampling
* Trend Analysis
* Seasonality Analysis
* Feature Engineering
* Lag Features
* Rolling Statistics
* Time-Based Data Splitting
* Machine Learning Model Comparison
* Hyperparameter Testing
* Linear Regression
* Random Forest
* LightGBM
* RMSE Optimization
* Model Validation
* Final Test Evaluation
* Data Leakage Prevention
* Python-based Data Analysis

---

# Conclusions

The objective of this project was to develop a machine learning solution capable of predicting the number of taxi orders for the next hour while achieving an RMSE no greater than 48.

The historical taxi order data was resampled into hourly intervals and analyzed to identify temporal patterns. The analysis revealed a general increasing trend and clear daily seasonality.

Several machine learning models were evaluated using different combinations of temporal features, lag variables, rolling means, and hyperparameters.

The main conclusions were:

1. The hourly resampling successfully transformed the original 10-minute data into a time series suitable for one-hour demand forecasting.

2. Preserving chronological order during train, validation, and test splitting prevented future information leakage.

3. Lag variables and rolling averages were effective for representing the recent dynamics of taxi demand.

4. Linear Regression achieved the best validation result with an RMSE of **30.72**.

5. The best feature configuration used **72 lag variables** and a **4-hour rolling mean**.

6. Random Forest achieved an RMSE of **31.04**, but did not outperform Linear Regression.

7. LightGBM obtained an RMSE of **48.03**, making it the weakest model among those evaluated.

8. After retraining the selected Linear Regression model using 90% of the available data, the final test evaluation produced an RMSE of **42.48**.

The final result meets the project requirement:

```text
Final Test RMSE = 42.48

Required RMSE ≤ 48

✓ OBJECTIVE ACHIEVED
```

This project demonstrates that, for this hourly taxi demand forecasting problem, a relatively simple Linear Regression model combined with well-designed temporal features, historical lags, and rolling statistics can outperform more complex machine learning models.

---

## Author

**Pablo Andrés Vázquez Calva**

Data Analyst | Data Science

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
