# ⚡ Power Grid Demand Forecasting: Production Pipeline

## Overview
This repository contains the production-ready pipeline (`PS.ipynb`) for forecasting hourly power demand. Predicting grid load across a decade of data requires more than just standard machine learning; it requires a model that understands long-term grid growth, short-term human behavior, and the physical limits of power sensors. 

This notebook processes raw historical data, algorithmically cleanses hardware errors, engineers physics-based features, and trains an XGBoost regression model to predict future demand.

## Pipeline Architecture

### 1. Data Integrity & Outlier Correction
Raw sensor data is inherently messy and prone to severe dropouts. To clean this without destroying legitimate low-demand periods (like holidays), we implemented a hybrid ML + Physics approach:
1. **Unsupervised Isolation Forest (PyOD):** We train an `IForest` algorithm, feeding it both the power demand and cyclical time features (hour/month sine and cosine). This allows the model to understand the contextual time of day and flag mathematically isolated anomalies (contamination = 0.5%).
2. **The Physical Failsafe:** Algorithms can sometimes over-flag tiny nighttime fluctuations. To prevent this, we apply a strict physical floor. An anomaly is only destroyed if the PyOD model flags it *and* the demand violently deviates from a 24-hour rolling median by more than 500 MW.

### 2. Smart Imputation Strategy
We repair missing data using a dual-strategy approach based on the size of the gap:
* **Seasonal Imputation (For large gaps):** To safely bridge days of lost history, we explicitly look for the exact same point in the weekly cycle (168 hours prior) to fill the gap, copying the physical shape of previous weeks without flattening the daily curve.
* **Mathematical Imputation (For micro-gaps):** For tiny 1-to-3 hour dropouts, we use localized linear interpolation to draw a smooth line between points, bridging the dropout where the daily cycle isn't lost.

### 3. Feature Engineering
We transform raw timestamps and weather data into signals that the algorithm can interpret as human behavior.

* **Cyclical Time:** We use sine and cosine transformations for hours and months so the model mathematically understands that 11:00 PM is right next to 12:00 AM, preserving the continuous loop of daily and seasonal regimes.
* **Streamlined Weather:** We streamline our dataset by dropping redundant weather metrics. We strictly rely on `apparent_temperature` because the physical "feels like" condition—which factors in humidity and wind—is what actually drives humans to crank up their ACs or heaters. By removing overlapping signals, we give our algorithm a cleaner view of human behavior.
* **Memory & Momentum:** We build short-term memory into our model so it understands exactly how power demand moves over time. By calculating the "deltas" (the mathematical difference between recent hours or previous days), we capture the grid's physical momentum. This teaches the algorithm to sense whether demand is currently accelerating into a heavy peak or rapidly cooling off.
* **The Macro Anchor:** A macro yearly indicator is used to tackle our biggest mathematical problem first: the massive 10-year growth of the power grid. It acts as a heavy anchor to establish the base demand for the current era, which perfectly sets the stage for our short-term momentum features to step in and fine-tune the exact hour-by-hour forecast.

### 4. Model Training & Forecasting
We utilize an **XGBoost Regressor** (`XGBRegressor`). Tree-based gradient boosting is uniquely suited for this task because it handles non-linear relationships (e.g., demand spikes at both extreme cold and extreme hot temperatures) and robustly manages the complex hierarchy between our macro-yearly trends and micro-hourly deltas.

### 5. Evaluation
The model's accuracy is evaluated using **MAPE (Mean Absolute Percentage Error)**, ensuring our error metric remains completely relative and interpretable regardless of the grid's baseline growth over the decade.

* **Final MAPE obtained = 1.770 %** 

## Setup & Execution

**Dependencies:**
```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
