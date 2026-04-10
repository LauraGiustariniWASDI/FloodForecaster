## Overview

The `flood_forecaster` processor is an advanced spatial machine learning pipeline that predicts and maps flood inundation risk at a high-resolution pixel level. It combines physical terrain characteristics derived from a Digital Elevation Model (DEM), Land Use/Land Cover (LULC) data, and multi-temporal satellite rainfall data (IMERG/GFS) to train and deploy highly optimized predictive models. 

The current architecture strictly separates the **Model Training** phase from the **Testing/Operational** phase, and introduces a robust physics engine and data augmentation to handle the inherent uncertainties of meteorological forecasts.

#### Key Features:

* **Land Use/Land Cover (LULC) Integration**: Automatically applies One-Hot Encoding to categorical LULC data (e.g., Built-Up, Trees, Bare soil), allowing the AI to calculate runoff potential based on impervious surfaces.
* **Physics Engine (Interaction Features)**: Forces the ML model to respect physical hydrology by mathematically combining terrain and rainfall into new features before training.
* **Data Augmentation (Jittering)**: Stress-tests the AI and prevents overfitting by tripling the training dataset. It applies a $\pm 20\%$ variance to rainfall volumes, teaching the model to remain stable even when weather forecasts are imprecise.
* **Sunny Day Failsafe**: Automatically bypasses the heavy ML computation and guarantees zero false alarms by instantly outputting a dry map if the maximum 24-hour rainfall is less than `1.0mm`.
* **Operational Freshness Safeguard**: When predicting future floods, the app automatically forces a cache refresh (`REPROCESS_ALL=True`) to guarantee the forecast uses the absolute latest GFS weather updates.
* **RAM-Safe Incremental Batch Training**: Bypasses traditional memory crashes by processing datasets in strict RAM-safe batches. The algorithm dynamically updates the saved `.joblib` model without "forgetting" past data.
* **Smart Ratio Downsampling**: Automatically balances class disparities by extracting 100% of flooded pixels and dynamically downsampling dry pixels to a 10:1 ratio, forcing the algorithm to learn sharp spatial boundaries without being drowned out by dry data.
* **Parquet Caching**: Extracted hydrological and meteorological data is saved as temporary `.parquet` files. Subsequent historical runs will intelligently skip heavy GDAL warping phases, drastically reducing compute time.


### Parameters

#### Required
* `BASENAME_FLOODMAP`: The common prefix string identifying the target flood maps in your workspace (e.g., `"PWThies"`).
* `SUFFIX_FLOODMAP`: The suffix string identifying the target flood maps (e.g., `"_flood.tif"`).
* `BASENAME_IMERG`: The common prefix string identifying the cumulative IMERG rainfall maps (e.g., `"Thies_Cumulative_"`).
* `DEM`: The exact filename of the Digital Elevation Model in your workspace to be used for terrain analysis (e.g., `"Thies_DEM15m.tif"`).
* `LULC_MAP`: The exact filename of the Land Use map (e.g., `"Thies_LULC.tif"`).

#### Operational, Batching & ML
* `OPERATIONAL` (defaults to `false`): Feature flag for real-time GFS forecasting deployment.
* `FORECAST_DATETIME`: The target future date and time for operational mode (e.g., `"2026-04-01 19:00"`).
* `TEST_DATE` (defaults to `""`): The exact historical date (YYYY-MM-DD) to test. If provided, the app enters Testing Mode.
* `START_MAP_INDEX` (defaults to `1`): The numerical index of the first map to process in the alphabetically-sorted workspace list.
* `END_MAP_INDEX` (defaults to `null`): The numerical index of the last map to process. Leave blank to process all the way to the end of the available list.
* `REPROCESS_ALL` (defaults to `false`): Set to `true` to force the app to overwrite existing `.parquet` cache files and rebuild the features.
* `ALGORITHM` (defaults to `"XGBoost"`): The machine learning algorithm to deploy. Accepts `"XGBoost"`, `"RF"`, or `"Random Forest"`.
* `TECHNIQUE` (defaults to `"regression"`): The modeling technique to use. Accepts `"regression"` or `"classification"`.
* `BASELINE_MODEL` (defaults to `""`): The filename of a pre-trained `.joblib` model in the workspace. 
* `SAVE_BASELINE_MODEL` (defaults to `false`): Set to `true` to export or overwrite the trained `.joblib` model to the workspace.
* `THRESHOLD` (defaults to `0.5`): A float used to split regression probabilities into a binary 0/1 map.

#### Hydrology
* `COMPUTE_TWI` (defaults to `false`): Set to `true` to calculate the Topographic Wetness Index from the DEM.
* `TWI_MAP` (defaults to `""`): The filename of a pre-computed TWI map in the workspace. 
* `COMPUTE_HAND` (defaults to `false`): Set to `true` to calculate Height Above Nearest Drainage from the DEM.
* `HAND_MAP` (defaults to `""`): The filename of a pre-computed HAND map in the workspace. 
* `MIN_ACC_VALUE_HAND` (defaults to `800`): The flow accumulation threshold used to define drainage channels for the HAND calculation.


### How to Run: Training, Testing, and Operations

**1. Training Mode (Building the Brain)**
To train the model on a batch of historical maps, configure your parameters as follows:
* Set `"TEST_DATE": ""` and `"OPERATIONAL": false`.
* Define your batch using `"START_MAP_INDEX"` and `"END_MAP_INDEX"`.
* Set `"SAVE_BASELINE_MODEL": true`. 
* *Result*: The app ingests the maps, calculates interaction features, applies data augmentation jittering, trains the AI, and saves the resulting model.

**2. Testing Mode (Historical Inference)**
To test the model's accuracy on a specific historical date and generate visual maps:
* Set `"TEST_DATE"` to your target date (e.g., `"2022-12-02"`).
* Provide the name of your trained model in `"BASELINE_MODEL"`.
* Set `"SAVE_BASELINE_MODEL": false`.
* *Result*: The app skips the training phase, extracts data exclusively for the test date, evaluates it against the model, and exports the predicted 2D GeoTIFF maps.

**3. Operational Mode (Future Forecasting)**
To predict upcoming flood risk using GFS meteorological forecasts:
* Set `"OPERATIONAL": true`.
* Set `"FORECAST_DATETIME"` to the target time (e.g., `"2026-04-01 19:00"`).
* Provide your trained model in `"BASELINE_MODEL"`.
* *Result*: The app automatically triggers GFS data cumulation (if missing), forces a fresh data extraction, bypasses training, and outputs probability heatmaps for the future date.


### JSON Sample

Operational Forecast Example
```json
{
  "BASENAME_FLOODMAP": "PWThies",
  "SUFFIX_FLOODMAP": "_flood.tif",
  "LIST_MAPS_WITH_FLOOD": "ListInputFloodMaps.txt",
  "BASENAME_IMERG": "Thies_Cumulative_",
  "DEM": "Thies_DEM15m.tif",
  "COMPUTE_TWI": false,
  "TWI_MAP": "Thies_DEM15m_TWI.tif",
  "COMPUTE_HAND": false,
  "HAND_MAP": "Thies_DEM15m_HAND.tif",
  "MIN_ACC_VALUE_HAND": 800,
  "LULC_MAP": "Thies_LULC.tif",
  "FILL_DEM": false,
  "ALGORITHM": "xgboost",
  "TECHNIQUE": "regression",
  "BASELINE_MODEL": "PWThies_XGB_v1_model.joblib",
  "SAVE_BASELINE_MODEL": false,
  "THRESHOLD": 0.5,
  "OPERATIONAL": true,
  "REPROCESS_ALL": false,
  "TEST_DATE": "",
  "FORECAST_DATETIME": "2026-04-01 19:00",
  "START_MAP_INDEX": 1,
  "END_MAP_INDEX": null
}
```
