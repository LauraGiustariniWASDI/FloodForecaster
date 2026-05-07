## Overview

The `flood_forecaster_zonal` processor is an advanced spatial machine learning pipeline that predicts and maps flood inundation risk. Unlike traditional pixel-based models, this version utilizes **Zonal Aggregation** (compressing landscapes into ~500m blocks) to evaluate macro-level community risk. It combines physical terrain characteristics derived from a Digital Elevation Model (DEM), Land Use Land Cover (LULC), and multi-temporal satellite rainfall data (IMERG or GFS) to train and deploy highly optimized, physics-aware predictive models.

#### Key Features:

* **Zonal Aggregation (Spatial Compression)**: Compresses raw 15m pixels into customizable zones (default ~495m). It uses a mathematical `'max'` aggregation for flood tracking, ensuring severe localized floods are never diluted, drastically reducing spatial noise and false alarms.
* **Physics-Aware Algorithm (Monotonic Constraints)**: Handcuffs the model to the laws of hydrology. It forces the XGBoost algorithm to recognize that more rain must always increase risk, while higher elevation must always decrease it, making "dry valley" hallucinations mathematically impossible.
* **Physically-Backed Failsafes**: Implements a strict 10.0mm 24-hour rainfall threshold. If a zone receives less than 10mm of rain, topographic interaction features are mathematically zeroed out, and the model is completely bypassed, saving compute time.
* **Smart Training via Hard Negative Mining**: Automatically extracts 100% of flooded zones and balances the batch with dry zones at a 10:1 ratio. It intentionally sorts and feeds the model "trick questions" (the deepest, wettest-looking dry valleys) so the model explicitly learns the boundary between safety and danger.
* **Data Augmentation (Jittering)**: Automatically triplicates the training data by creating ±20% meteorological variance scenarios, stress-testing the model against erratic weather patterns.
* **Operational Forecast Mode**: Instantly toggle from historical testing to future forecasting. The app automatically fetches and cumulates real-time GFS weather data for a targeted future datetime.
* **Parquet Caching**: Extracted hydrological and meteorological data is securely saved as temporary `.parquet` files using the `fastparquet` engine. Subsequent runs intelligently skip heavy GDAL warping phases, and integrated garbage collection prevents server RAM crashes, drastically reducing compute time.

### Parameters

#### Required

* `BASENAME_FLOODMAP`: The common prefix string identifying the target flood maps in your workspace (e.g., `"PWThies"`).
* `SUFFIX_FLOODMAP`: The suffix string identifying the target flood maps (e.g., `"_flood.tif"`).
* `BASENAME_IMERG`: The common prefix string identifying the cumulative rainfall maps (e.g., `"Thies_Cumulative_"`).
* `DEM`: The exact filename of the Digital Elevation Model in your workspace (e.g., `"Thies_DEM15m.tif"`).

#### Operational, Batching & ML
* `ZONE_SIZE_PIXELS` (defaults to `33`): Defines the size of the aggregation block. (e.g., 33 pixels * 15m resolution = 495m zones).
* `TEST_DATE` (defaults to `""`): The exact date (YYYY-MM-DD) to test. If provided, the app enters **Testing Mode**. If blank, the app enters **Training Mode**.
* `OPERATIONAL` (defaults to `false`): Set to `true` to activate real-time forecasting. The app will fetch GFS data instead of historical IMERG.
* `FORECAST_DATETIME` (defaults to `""`): Required if `OPERATIONAL` is true. The exact target time for the forecast (e.g., `"2026-04-12 12:00"`).
* `REPROCESS_ALL` (defaults to `false`): Set to `true` to force the app to overwrite existing Parquet cache files and re-warp the raw GeoTIFFs. (Automatically set to `true` in Operational mode, or when recovering from a corrupted cache).
* `START_MAP_INDEX` (defaults to `1`): The numerical index of the first map to process in the workspace list.
* `END_MAP_INDEX` (defaults to `""` / `null`): The numerical index of the last map to process. 
* `ALGORITHM` (defaults to `"XGBoost"`): The machine learning algorithm to deploy. Accepts `"XGBoost"`, `"RF"`, or `"Random Forest"`.
* `BASELINE_MODEL` (defaults to `""`): The filename of a pre-trained `.joblib` model. Required for Testing/Operational Mode, and used in Training Mode to incrementally update an existing brain.
* `SAVE_BASELINE_MODEL` (defaults to `false`): Set to `true` to export the trained `.joblib` model to the workspace.
* `THRESHOLD` (defaults to `0.25`): The probability threshold used specifically for generating confusion matrix metrics in the JSON payload. 
* `LIST_MAPS_WITH_FLOOD` (defaults to `""`): The filename of a text list specifying exact flood maps to use.

#### Hydrology & LULC
* `COMPUTE_TWI` (defaults to `false`): Set to `true` to calculate the Topographic Wetness Index from the DEM.
* `TWI_MAP` (defaults to `""`): The filename of a pre-computed TWI map in the workspace. 
* `COMPUTE_HAND` (defaults to `false`): Set to `true` to calculate Height Above Nearest Drainage from the DEM.
* `HAND_MAP` (defaults to `""`): The filename of a pre-computed HAND map in the workspace. 
* `MIN_ACC_VALUE_HAND` (defaults to `200`): The flow accumulation threshold used to define drainage channels.
* `FILL_DEM` (defaults to `true`): Set to `true` to fill sinks in the DEM prior to computing hydrology features.
* `LULC_MAP` (defaults to `"Thies_LULC.tif"`): The filename of the Land Use Land Cover map. The app automatically performs one-hot encoding on ESA standard classes.

### How to Run: Training vs. Testing

**1. Training Mode (Building the Brain)**
* Set `"TEST_DATE": ""` and `"OPERATIONAL": false`.
* Set `"SAVE_BASELINE_MODEL": true`. 
* *Result*: The app ingests the historical maps, trains the physics-aware model on the aggregated zones, and saves the resulting model as `{BASENAME_FLOODMAP}_zonal_baseline_model.joblib`. 

**2. Historical Testing Mode (Inference & Evaluation)**
* Set `"TEST_DATE"` to your target date (e.g., `"2020-09-12"`).
* Provide the name of your trained model in `"BASELINE_MODEL"`.
* *Result*: The app skips training, extracts data exclusively for the test date, evaluates it, and exports a 2D probability heatmap alongside performance metrics.

**3. Operational Forecast Mode**
* Set `"OPERATIONAL": true` and provide a `"FORECAST_DATETIME"`.
* Provide your `"BASELINE_MODEL"`.
* *Result*: The app reaches out to the GFS Cumulator, builds the future meteorological arrays, and outputs the predictive Zonal Heatmap for the target datetime.

### Output Files:

**Payload Data**: 
* **Feature Importance**: A dictionary ranking how heavily the model relied on each input variable.
* **Test Set Metrics (Threshold X)**: Generates the Confusion Matrix results alongside advanced operational scores (**Precision, Recall, F1-Score**). *Only generated during Historical Testing Mode*.

**Workspace Files**:
* **Cached Model**: `{BASENAME_FLOODMAP}_zonal_baseline_model.joblib`
* **Predicted Flood Map**: A structured 2D array projected back to the original map boundaries: `{BASENAME}_{DATE}_ZonalFloatFlood.tif` (Float32 Probabilities).
* **Hydrology Maps**: Newly generated TWI/HAND maps (if requested).

### JSON Sample

Operational Forecast Example
```json
{
  "BASENAME_FLOODMAP": "PWThies",
  "SUFFIX_FLOODMAP": "_flood.tif",
  "BASENAME_IMERG": "Thies_Cumulative_",
  "DEM": "Thies_DEM15m.tif",
  "COMPUTE_TWI": false,
  "TWI_MAP": "Thies_DEM15m_TWI.tif",
  "COMPUTE_HAND": false,
  "HAND_MAP": "Thies_DEM15m_HAND.tif",
  "MIN_ACC_VALUE_HAND": 800,
  "LULC_MAP": "Thies_LULC.tif",
  "FILL_DEM": false,
  "ZONE_SIZE_PIXELS": 33,
  "ALGORITHM": "xgboost",
  "THRESHOLD": 0.25,
  "LIST_MAPS_WITH_FLOOD": "ListInputFloodMaps.txt",
  "START_MAP_INDEX": 1,
  "END_MAP_INDEX": null,
  "BASELINE_MODEL": "PWThies_zonal_baseline_model.joblib",
  "SAVE_BASELINE_MODEL": false,
  "OPERATIONAL": true,
  "REPROCESS_ALL": false,
  "FORECAST_DATETIME": "2020-09-12 19:00",
  "TEST_DATE": ""
}
```
```