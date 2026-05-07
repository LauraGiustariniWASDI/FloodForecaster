import wasdi
import re
import os
from datetime import datetime
from osgeo import gdal, osr
import numpy as np
import pandas as pd

from whitebox.whitebox_tools import WhiteboxTools
from pysheds.grid import Grid
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import joblib
import glob


def run():

    wasdi.wasdiLog("START: Flood Forecaster Zonal v.2.0.5")
    aoPayload = {}

    try:
        sBasenamefloodmap = wasdi.getParameter("BASENAME_FLOODMAP", "PWThies") 
        sSuffixfloodmap = wasdi.getParameter("SUFFIX_FLOODMAP", "_flood.tif") 
        sBasenameimerg = wasdi.getParameter("BASENAME_IMERG", "Thies_Cumulative_") 
        
        # ZONAL AGGREGATION PARAMETER
        iZoneSize = int(wasdi.getParameter("ZONE_SIZE_PIXELS", 33)) # 33 pixels * 15m = ~495m zones
        
        # HYDROLOGY PARAMETERS
        sDem = wasdi.getParameter("DEM", "Thies_DEM30m.tif")
        bFillDem = wasdi.getParameter("FILL_DEM", True)
        bComputeTwi = wasdi.getParameter("COMPUTE_TWI", False)
        sTwiMap = wasdi.getParameter("TWI_MAP", "")
        bComputeHand = wasdi.getParameter("COMPUTE_HAND", False)
        sHandMap = wasdi.getParameter("HAND_MAP", "")
        iMinAccValueRaw = wasdi.getParameter("MIN_ACC_VALUE_HAND", 200)
        iMinAccValue = int(iMinAccValueRaw) if iMinAccValueRaw else 200
        
        # LULC PARAMETER
        sLulcMap = wasdi.getParameter("LULC_MAP", "Thies_LULC.tif")
        bUseLulc = (sLulcMap != "")
        
        # OPERATIONAL, BATCHING & ML PARAMETERS
        bOperationalMode = wasdi.getParameter("OPERATIONAL", False)
        bReprocessAll = wasdi.getParameter("REPROCESS_ALL", False)        
        if bOperationalMode:    # OPERATIONAL FRESHNESS SAFEGUARD
            bReprocessAll = True
        sTestDate = str(wasdi.getParameter("TEST_DATE", "")).strip() 
        sForecastDateTime = str(wasdi.getParameter("FORECAST_DATETIME", "")).strip()
        
        # BATCHING PARAMETERS
        iStartMapRaw = wasdi.getParameter("START_MAP_INDEX", 1)
        iEndMapRaw = wasdi.getParameter("END_MAP_INDEX", "")
        iStartMap = max(0, int(iStartMapRaw) - 1) if iStartMapRaw not in ["", None] else 0
        iEndMap = int(iEndMapRaw) if iEndMapRaw not in ["", None] else None
        
        fThresholdRaw = wasdi.getParameter("THRESHOLD", None)
        fThreshold = float(fThresholdRaw) if fThresholdRaw not in ["", None] else 0.5
        sListMapsWithFloodTxt = wasdi.getParameter("LIST_MAPS_WITH_FLOOD", "") 
        sAlgorithmRaw = wasdi.getParameter("ALGORITHM", "XGBoost")
        sAlgorithm = str(sAlgorithmRaw).strip().lower() if sAlgorithmRaw else "xgboost"
        sModelBaselineJoblib = wasdi.getParameter("BASELINE_MODEL", "")
        bSaveBaselineModel = wasdi.getParameter("SAVE_BASELINE_MODEL", False)

        aoPayload["INPUTS"] = wasdi.getParametersDict()
        wasdi.setPayload(aoPayload)
        wasdi.wasdiLog("All input parameters read")
        wasdi.updateProgressPerc(5) 

        # --- 1. COMPUTE HYDROLOGICAL VARIABLES ---
        if bComputeTwi:
            wasdi.wasdiLog("Computing Topographic Wetness Index (TWI)...")
            wbt = WhiteboxTools()
            if bFillDem:
                sFilledDem = sDem.replace(".tif", "_filled.tif")
                wbt.fill_depressions(dem=wasdi.getPath(sDem), output=wasdi.getPath(sFilledDem))
            else:
                sFilledDem = sDem

            sSlope = sDem.replace(".tif", "_slope.tif")
            wbt.slope(dem=wasdi.getPath(sFilledDem), output=wasdi.getPath(sSlope), units="degrees")

            sSpecContrArea = sDem.replace(".tif", "_sca.tif")
            wbt.d8_flow_accumulation(wasdi.getPath(sFilledDem), wasdi.getPath(sSpecContrArea), out_type="specific contributing area")

            sTWI = sDem.replace(".tif", "_TWI.tif")
            wbt.wetness_index(sca=wasdi.getPath(sSpecContrArea), slope=wasdi.getPath(sSlope), output=wasdi.getPath(sTWI))
            wasdi.addFileToWASDI(sTWI)
            sTwiMap = sTWI 
        elif sTwiMap != "":
            wasdi.wasdiLog(f"Using pre-computed TWI map: {sTwiMap}")

        if bComputeHand:
            wasdi.wasdiLog("Computing Height Above Nearest Drainage (HAND)...")
            oGrid = Grid.from_raster(wasdi.getPath(sDem))
            oDem = oGrid.read_raster(wasdi.getPath(sDem))
            oInflatedDem = oGrid.resolve_flats(oDem)
            oFlowDir = oGrid.flowdir(oInflatedDem)
            oAcc = oGrid.accumulation(oFlowDir)
            hand = oGrid.compute_hand(oFlowDir, oDem, oAcc > iMinAccValue)
            
            sHand = sDem.replace(".tif", "_HAND.tif")
            oGrid.to_raster(hand, wasdi.getPath(sHand))
            wasdi.addFileToWASDI(sHand)
            sHandMap = sHand 
        elif sHandMap != "":
            wasdi.wasdiLog(f"Using pre-computed HAND map: {sHandMap}")

        wasdi.updateProgressPerc(10)

        # --- 2. SELECT MAPS & APPLY BATCH SLICING ---
        wasdi.wasdiLog("Start selecting maps from workspace...")
        asCurrentFiles = wasdi.getProductsByActiveWorkspace()

        if sListMapsWithFloodTxt == "":
            asAllAvailableMaps = []
            for sCurrentFile in asCurrentFiles:
                if sBasenamefloodmap in sCurrentFile and sSuffixfloodmap in sCurrentFile:
                    sCurrentFilePath = wasdi.getPath(sCurrentFile)
                    oFloodMap = gdal.Open(sCurrentFilePath)
                    afFloodMapArray = np.array(oFloodMap.GetRasterBand(1).ReadAsArray())
                    if np.max(afFloodMapArray) == 3: 
                        asAllAvailableMaps.append(sCurrentFile)
            asAllAvailableMaps.sort()
            asFloodMaps = asAllAvailableMaps[iStartMap:iEndMap]
            iEndLog = iEndMap if iEndMap else len(asAllAvailableMaps)
            wasdi.wasdiLog(f"Batch Filter Applied: Selected {len(asFloodMaps)} maps.")

        else:
            with open(wasdi.getPath(sListMapsWithFloodTxt), "r") as f:
                asAllAvailableMaps = f.read().splitlines()
                asAllAvailableMaps.sort() 
                asFloodMaps = asAllAvailableMaps[iStartMap:iEndMap]
            wasdi.wasdiLog(f"Batch Filter Applied: Selected {len(asFloodMaps)} maps from text list.")

        # --- 2.5 APPLY TEST/FORECAST & OPERATIONAL FILTER ---
        sGFSTargetTime = ""
        bUseGFS = False
        asFutureRainMaps = []
        asRainCumulVals = ["1hr", "3hr", "6hr", "12hr", "24hr"]

        if bOperationalMode:
            if sForecastDateTime == "": raise Exception("CRITICAL ERROR: OPERATIONAL mode requires FORECAST_DATETIME.")
            bUseGFS = True
            sGFSTargetTime = sForecastDateTime
            try:
                dtTarget = datetime.strptime(sGFSTargetTime, '%Y-%m-%d %H:%M')
                sTargetDateDigits = dtTarget.strftime("%Y%m%d")
                sTargetTimeDigits = dtTarget.strftime("%H%M")
            except ValueError:
                sTargetDateDigits = sGFSTargetTime.replace("-", "").replace(" ", "").replace(":", "")
                sTargetTimeDigits = ""

            sFullTargetDigits = sTargetDateDigits + sTargetTimeDigits
            
            asPotentialMaps = []
            for sFile in asCurrentFiles:
                if sBasenameimerg in sFile:
                    sFileDigits = "".join([c for c in sFile if c.isdigit()])
                    if sFullTargetDigits in sFileDigits or (sTargetTimeDigits == "" and sTargetDateDigits in sFileDigits):
                        asPotentialMaps.append(sFile)
            
            for sPeriod in asRainCumulVals:
                for sFile in asPotentialMaps:
                    if sPeriod in sFile and sFile not in asFutureRainMaps:
                        asFutureRainMaps.append(sFile)
                        break 
            
            # Trigger Cumulator if ANY of the 5 periods are missing
            if len(asFutureRainMaps) < 5:
                wasdi.wasdiLog(f"Missing required GFS/IMERG cumulates for exact datetime {sGFSTargetTime}. Triggering Cumulator App...")
                
                # Extract BBOX from DEM to define the processing area
                oDemData = gdal.Open(wasdi.getPath(sDem))
                fUlx, fXres, fXskew, fUly, fYskew, fYres = oDemData.GetGeoTransform()
                iRows, iCols = oDemData.RasterYSize, oDemData.RasterXSize
                fLrx = fUlx + (iCols * fXres)
                fLry = fUly + (iRows * fYres)
                oDemData = None
                
                # Build the payload for the external processor
                aoGFSDict = {
                    "BASE_NAME": sBasenamefloodmap,
                    "BBOX": {
                        "northEast": {"lat": fUly, "lng": fLrx},
                        "southWest": {"lat": fLry, "lng": fUlx}
                    },
                    "TARGET_DATETIME": sGFSTargetTime,
                    "CUMULATION_PERIODS": "1hr,3hr,6hr,12hr,24hr",
                    "DELETE": True
                }
                
                # Execute the Cumulator and wait for completion
                sGFSCumulatorId = wasdi.executeProcessor("gfs_precipitation_cumulator", aoGFSDict)
                wasdi.wasdiLog(f"Waiting for Cumulator App (Process ID: {sGFSCumulatorId})...")
                wasdi.waitProcesses([sGFSCumulatorId])
                
                # Retrieve the newly generated files from the processor's payload
                aoGFSPayload = wasdi.getProcessorPayloadAsJson(sGFSCumulatorId)
                asFutureRainMaps = aoGFSPayload.get("OUTPUTS", [])
                wasdi.wasdiLog(f"Cumulator finished. Retrieved {len(asFutureRainMaps)} forecast maps.")
            else:
                wasdi.wasdiLog(f"Found all 5 required cumulative maps in workspace for {sGFSTargetTime}. Skipping Cumulator app.")
    
            if len(asAllAvailableMaps) == 0: raise Exception("CRITICAL ERROR: No historical map found for spatial template.")
            asFloodMaps = [asAllAvailableMaps[0]] 

        elif sTestDate != "":
            asFloodMaps = [f for f in asFloodMaps if sTestDate in f]
            if len(asFloodMaps) == 0: raise Exception(f"CRITICAL ERROR: No map found matching {sTestDate}.")
            if sModelBaselineJoblib == "": raise Exception("CRITICAL ERROR: TEST_DATE requires a BASELINE_MODEL.")

        asIMERGMaps = [f for f in asCurrentFiles if sBasenameimerg in f]
        aoPayload["Total Flood Maps Processing"] = len(asFloodMaps) if not bOperationalMode else 1
        wasdi.setPayload(aoPayload)
        wasdi.updateProgressPerc(20)

        # --- 3. BUILD DATAFRAME (ITERATIVE PARQUET METHOD WITH ZONAL AGGREGATION) ---
        if len(asFloodMaps) > 0:
            wasdi.wasdiLog("Building Zonal Tabular Dataset iteratively on disk via Parquet...")
        
        bUseTwi = (sTwiMap != "")
        bUseHand = (sHandMap != "")

        for iCounter, sFloodMap in enumerate(asFloodMaps):
            wasdi.wasdiLog(f"Processing Spatial Grid: {sFloodMap}")

            for sRequiredFile in [sFloodMap, sDem, sTwiMap, sHandMap]:
                if not os.path.exists(wasdi.getPath(sRequiredFile)):
                    raise Exception(f"CRITICAL ERROR: The file '{sRequiredFile}' is missing!")
            
            iCurrentProgress = 20 + int(((iCounter) / len(asFloodMaps)) * 30)
            wasdi.updateProgressPerc(iCurrentProgress)
            
            sMapDateRaw = re.search(r'\d{4}-\d{2}-\d{2}', sFloodMap).group()
            sDate = sGFSTargetTime.split(" ")[0] if bOperationalMode else sMapDateRaw
            sParquetName = wasdi.getPath(f"temp_zonal_map_data_{sDate}.parquet")
            
            if os.path.exists(sParquetName) and not bReprocessAll:
                wasdi.wasdiLog(f"Parquet already exists for {sDate}. Skipping warp phase.")
                continue
            elif os.path.exists(sParquetName) and bReprocessAll:
                wasdi.wasdiLog(f"REPROCESS_ALL active: Overwriting Parquet for {sDate}.")

            oFloodMap = gdal.Open(wasdi.getPath(sFloodMap))
            if oFloodMap is None: continue

            fUlx, fXres, fXskew, fUly, fYskew, fYres = oFloodMap.GetGeoTransform()
            iRows, iCols = oFloodMap.RasterYSize, oFloodMap.RasterXSize
            
            afColGrid0, afRowGrid0 = np.meshgrid(np.arange(iCols), np.arange(iRows))

            afFloodMap1D = np.array(oFloodMap.GetRasterBand(1).ReadAsArray()).ravel()
            valid_mask = (afFloodMap1D != 0) & (afFloodMap1D != 2)
            if np.sum(valid_mask) == 0: continue
                
            dict_valid_data = {
                'FloodedNonFlooded': afFloodMap1D[valid_mask],
                'PosX': afColGrid0.ravel()[valid_mask], 'PosY': afRowGrid0.ravel()[valid_mask]
            }
            del afColGrid0, afRowGrid0
            
            afBbox = [fUlx, fUly + (iRows * fYres), fUlx + (iCols * fXres), fUly]
            sDestProj = oFloodMap.GetProjection()
            aoWarpOptions = gdal.WarpOptions(srcSRS=sDestProj, dstSRS=sDestProj, xRes=fXres, yRes=abs(fYres), outputBounds=afBbox, format="MEM")

            oDem = gdal.Open(wasdi.getPath(sDem))
            oWarpedDem = gdal.Warp("", oDem, options=aoWarpOptions)
            dict_valid_data['DEM'] = np.array(oWarpedDem.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
            oDem = None; oWarpedDem = None 

            if bUseTwi: 
                oTWI = gdal.Open(wasdi.getPath(sTwiMap))
                oWarpedTWI = gdal.Warp("", oTWI, options=aoWarpOptions)
                dict_valid_data['TWI'] = np.array(oWarpedTWI.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
                oTWI = None; oWarpedTWI = None
                
            if bUseHand: 
                oHAND = gdal.Open(wasdi.getPath(sHandMap))
                oWarpedHAND = gdal.Warp("", oHAND, options=aoWarpOptions)
                dict_valid_data['HAND'] = np.array(oWarpedHAND.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
                oHAND = None; oWarpedHAND = None

            if bUseLulc: 
                oLULC = gdal.Open(wasdi.getPath(sLulcMap))
                aoWarpOptionsLULC = gdal.WarpOptions(srcSRS=sDestProj, dstSRS=sDestProj, xRes=fXres, yRes=abs(fYres), outputBounds=afBbox, format="MEM", resampleAlg=gdal.GRA_NearestNeighbour)
                oWarpedLULC = gdal.Warp("", oLULC, options=aoWarpOptionsLULC)
                afRawLULC = np.array(oWarpedLULC.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
                
                dict_valid_data['LULC_Trees'] = (afRawLULC == 10).astype(int)
                dict_valid_data['LULC_ShrubGrass'] = np.isin(afRawLULC, [20, 30]).astype(int)
                dict_valid_data['LULC_Crop'] = (afRawLULC == 40).astype(int)
                dict_valid_data['LULC_BuiltUp'] = (afRawLULC == 50).astype(int)
                dict_valid_data['LULC_Bare'] = (afRawLULC == 60).astype(int)
                dict_valid_data['LULC_WaterWetland'] = np.isin(afRawLULC, [80, 90, 95]).astype(int)
                oLULC = None; oWarpedLULC = None; del afRawLULC

            # Meteo Extraction
            if bOperationalMode:
                for sRainMap in asFutureRainMaps:
                    for sCumulVal in asRainCumulVals:
                        if sCumulVal in sRainMap:
                            oRainData = gdal.Open(wasdi.getPath(sRainMap))
                            oWarpedRain = gdal.Warp("", oRainData, options=aoWarpOptions)
                            dict_valid_data["RainCum_" + sCumulVal] = np.array(oWarpedRain.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
                            oRainData = None; oWarpedRain = None
            else:
                sIMERGDate = datetime.strptime(sDate, '%Y-%m-%d').strftime("%Y%m%d")
                asDateIMERGMaps = [m for m in asIMERGMaps if f"-{sIMERGDate}" in m]
                for sIMERGMap in asDateIMERGMaps:
                    for sCumulVal in asRainCumulVals:
                        if sCumulVal in sIMERGMap:
                            oIMERG = gdal.Open(wasdi.getPath(sIMERGMap))
                            oWarpedIMERG = gdal.Warp("", oIMERG, options=aoWarpOptions)
                            dict_valid_data["RainCum_" + sCumulVal] = np.array(oWarpedIMERG.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
                            oIMERG = None; oWarpedIMERG = None

            dfFloodInLoop = pd.DataFrame(dict_valid_data)
            del dict_valid_data, valid_mask
            
            if bUseTwi: dfFloodInLoop['TWI'] = dfFloodInLoop['TWI'].fillna(0)
            if bUseHand: dfFloodInLoop['HAND'] = dfFloodInLoop['HAND'].fillna(9999)


            # INTERACTION FEATURES WITH GATING
            if 'RainCum_24hr' in dfFloodInLoop.columns:
                # The Gate: 1 if severe storm, 0 if dry/light rain
                dfFloodInLoop['Is_Storm_Active'] = (dfFloodInLoop['RainCum_24hr'] >= 10.0).astype(int)
                
                if bUseHand:
                    dfFloodInLoop['Interaction_Rain24_HAND'] = (dfFloodInLoop['RainCum_24hr'] / (dfFloodInLoop['HAND'] + 1.0)) * dfFloodInLoop['Is_Storm_Active']
                if bUseTwi:
                    dfFloodInLoop['Interaction_Rain24_TWI'] = (dfFloodInLoop['RainCum_24hr'] * dfFloodInLoop['TWI']) * dfFloodInLoop['Is_Storm_Active']

            dfFloodInLoop = dfFloodInLoop.dropna()
            
            # Binary translation before fractional mean
            dfFloodInLoop.loc[dfFloodInLoop["FloodedNonFlooded"] == 1, "FloodedNonFlooded"] = 0
            dfFloodInLoop.loc[dfFloodInLoop["FloodedNonFlooded"] == 3, "FloodedNonFlooded"] = 1

            # --- ZONAL AGGREGATION BLOCK ---
            dfFloodInLoop['Zone_X'] = dfFloodInLoop['PosX'] // iZoneSize
            dfFloodInLoop['Zone_Y'] = dfFloodInLoop['PosY'] // iZoneSize
            
            dict_agg = {
                'FloodedNonFlooded': 'max', # <--- CHANGE THIS FROM 'mean' TO 'max'
                'DEM': 'mean',
                'PosX': 'min', 
                'PosY': 'min'
            }
            
            if bUseTwi: dict_agg['TWI'] = 'mean'; dict_agg['Interaction_Rain24_TWI'] = 'mean'
            if bUseHand: dict_agg['HAND'] = 'min'; dict_agg['Interaction_Rain24_HAND'] = 'mean'
            if bUseLulc:
                for l_col in ['LULC_Trees', 'LULC_ShrubGrass', 'LULC_Crop', 'LULC_BuiltUp', 'LULC_Bare', 'LULC_WaterWetland']:
                    dict_agg[l_col] = 'mean' # Converts One-Hot to Percentage
            for sCumulVal in asRainCumulVals:
                if "RainCum_" + sCumulVal in dfFloodInLoop.columns:
                    dict_agg["RainCum_" + sCumulVal] = 'mean'
            
            df_zonal = dfFloodInLoop.groupby(['Zone_X', 'Zone_Y']).agg(dict_agg).reset_index()
            df_zonal['Date'] = sDate
            
            df_zonal.to_parquet(sParquetName, engine='fastparquet', compression='snappy')
            del dfFloodInLoop, df_zonal

        wasdi.updateProgressPerc(50) 

        # --- 4. DATA ROUTING (TRAINING vs TESTING) ---
        asAllParquetFiles = []
        for sMap in asFloodMaps:
            sMapDate = sForecastDateTime.split(" ")[0] if bOperationalMode else re.search(r'\d{4}-\d{2}-\d{2}', sMap).group()
            sExpectedPath = wasdi.getPath(f"temp_zonal_map_data_{sMapDate}.parquet")
            if os.path.exists(sExpectedPath): asAllParquetFiles.append(sExpectedPath)
                
        asTrainParquetFiles = []
        asTestParquetFiles = []

        if sTestDate == "" and not bOperationalMode:
            asTrainParquetFiles = asAllParquetFiles
        else:
            asTestParquetFiles = asAllParquetFiles

        wasdi.updateProgressPerc(55) 
        
        # --- 5. MODEL INITIALIZATION & TRAINING WITH JITTERING ---
        bIsXGBoost = sAlgorithm not in ["random forest", "randomforest", "rf"]
        BATCH_SIZE = 30 
        
        asFeatureNames = ['DEM']
        if bUseTwi: 
            asFeatureNames.append('TWI')
            asFeatureNames.append('Interaction_Rain24_TWI')
        if bUseHand: 
            asFeatureNames.append('HAND')
            asFeatureNames.append('Interaction_Rain24_HAND')
        if bUseLulc: 
            asFeatureNames.extend(['LULC_Trees', 'LULC_ShrubGrass', 'LULC_Crop', 'LULC_BuiltUp', 'LULC_Bare', 'LULC_WaterWetland'])
        asFeatureNames.extend(["RainCum_1hr", "RainCum_3hr", "RainCum_6hr", "RainCum_12hr", "RainCum_24hr"])

        if sModelBaselineJoblib == "":
            wasdi.wasdiLog(f"Initializing new {sAlgorithm} regression model with Zonal Aggregation...")
            if not bIsXGBoost:
                iTotalBatches = max(1, (len(asTrainParquetFiles) + BATCH_SIZE - 1) // BATCH_SIZE)
                iTreesPerBatch = max(5, int(200 / iTotalBatches)) 
                params = {"n_estimators": 0, "max_features": "sqrt", "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 4, "bootstrap": False, "warm_start": True, "random_state": 42, "n_jobs": -1}
                oModel = RandomForestRegressor(**params)
            else:
                params = {"n_estimators": 50, "max_depth": 7, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1}
                
                # MONOTONIC CONSTRAINTS
                dict_constraints = {}
                for sFeat in asFeatureNames:
                    if "RainCum" in sFeat or "TWI" in sFeat or "Interaction" in sFeat:
                        dict_constraints[sFeat] = 1   # Positive: Value goes up -> Risk goes up
                    elif "HAND" in sFeat or "DEM" in sFeat:
                        dict_constraints[sFeat] = -1  # Negative: Value goes up -> Risk goes down
                    else:
                        dict_constraints[sFeat] = 0   # LULC: Neutral constraint
                params["monotone_constraints"] = dict_constraints
                
                oModel = XGBRegressor(**params)
        else:
            oModel = joblib.load(wasdi.getPath(sModelBaselineJoblib))
            if not bIsXGBoost: iTreesPerBatch = 10 

        if len(asTrainParquetFiles) > 0:
            columns_to_drop = ['FloodedNonFlooded', 'Date', 'PosX', 'PosY', 'Zone_X', 'Zone_Y']
            iTotalTrainingBatches = (len(asTrainParquetFiles) + BATCH_SIZE - 1) // BATCH_SIZE
            iBatchCounter = 0
            
            for i in range(0, len(asTrainParquetFiles), BATCH_SIZE):
                wasdi.updateProgressPerc(55 + int((iBatchCounter / iTotalTrainingBatches) * 25))
                iBatchCounter += 1
                asBatchFiles = asTrainParquetFiles[i:i+BATCH_SIZE]
                
                asDataFramesToConcat = []
                for sFile in asBatchFiles:
                    df_Map = pd.read_parquet(sFile)                    
                    
                    # Extract ALL zones without any per-file limits
                    df_flooded = df_Map[df_Map['FloodedNonFlooded'] > 0.0]
                    df_dry = df_Map[df_Map['FloodedNonFlooded'] == 0.0]
                        
                    asDataFramesToConcat.append(df_flooded)
                    asDataFramesToConcat.append(df_dry)
                    del df_Map
                    
                df_batch = pd.concat(asDataFramesToConcat, ignore_index=True)
                del asDataFramesToConcat
                
                # MAXIMIZED GLOBAL BATCH BALANCING
                df_batch_flooded = df_batch[df_batch['FloodedNonFlooded'] > 0.0]
                df_batch_dry = df_batch[df_batch['FloodedNonFlooded'] == 0.0]
                
                # We keep 100% of the precious flooded zones.
                if len(df_batch_flooded) > 0:
                    iTargetDryCount = len(df_batch_flooded) * 10 
                    
                    # HARD NEGATIVE MINING
                    if len(df_batch_dry) > iTargetDryCount: 
                        # Sort to put the most physically vulnerable dry zones at the top
                        if 'TWI' in df_batch_dry.columns:
                            df_batch_dry = df_batch_dry.sort_values(by='TWI', ascending=False)
                        elif 'HAND' in df_batch_dry.columns:
                            df_batch_dry = df_batch_dry.sort_values(by='HAND', ascending=True)
                        else:
                            df_batch_dry = df_batch_dry.sort_values(by='DEM', ascending=True)

                        iHardCount = int(iTargetDryCount * 0.3) # 30% trick questions
                        iRandomCount = iTargetDryCount - iHardCount # 70% normal background
                        
                        df_hard = df_batch_dry.head(iHardCount)
                        df_random = df_batch_dry.iloc[iHardCount:].sample(n=iRandomCount, random_state=42)
                        df_batch_dry = pd.concat([df_hard, df_random], ignore_index=True)
                        
                df_batch = pd.concat([df_batch_flooded, df_batch_dry], ignore_index=True)
                
                X_base = df_batch.drop(columns_to_drop, axis=1, errors='ignore')[asFeatureNames].astype('float64')
                y_base = df_batch[['FloodedNonFlooded']]
                
                # DATA AUGMENTATION (JITTERING)
                asRainAndInteractionCols = [c for c in X_base.columns if "RainCum" in c or "Interaction" in c]
                
                X_down = X_base.copy()
                X_down[asRainAndInteractionCols] = X_down[asRainAndInteractionCols] * 0.8
                
                X_up = X_base.copy()
                X_up[asRainAndInteractionCols] = X_up[asRainAndInteractionCols] * 1.2
                
                X_batch_final = pd.concat([X_base, X_down, X_up], ignore_index=True)
                y_batch_final = pd.concat([y_base, y_base, y_base], ignore_index=True)
                
                # Shuffle the final jittered dataset
                idx_shuffle = np.random.permutation(len(X_batch_final))
                X_batch_final = X_batch_final.iloc[idx_shuffle]
                y_batch_final = y_batch_final.iloc[idx_shuffle]

                if not bIsXGBoost:
                    oModel.n_estimators += iTreesPerBatch
                    oModel.fit(X_batch_final, np.ravel(y_batch_final))
                else:
                    if i == 0 and sModelBaselineJoblib == "":
                        oModel.fit(X_batch_final, np.ravel(y_batch_final)) 
                    else:
                        oModel.fit(X_batch_final, np.ravel(y_batch_final), xgb_model=oModel.get_booster()) 
                
                wasdi.wasdiLog(f"Batch {int(i/BATCH_SIZE)+1} trained on {len(X_batch_final):,} Augmented ZONES.")
                del df_batch, X_base, y_base, X_down, X_up, X_batch_final, y_batch_final 

            if bSaveBaselineModel:
                sOutModelName = f"{sBasenamefloodmap}_zonal_baseline_model.joblib"
                joblib.dump(oModel, wasdi.getPath(sOutModelName))
                wasdi.addFileToWASDI(sOutModelName)
        else:
            wasdi.wasdiLog("Training skipped (Test Mode Active).")

        wasdi.updateProgressPerc(80) 

        # --- 6. ITERATIVE EVALUATION FOR PAYLOAD METRICS ---
        try:
            feature_importance = oModel.feature_importances_
            sorted_idx = np.argsort(feature_importance)[::-1]
            aoPayload["Feature Importance"] = {asFeatureNames[idx]: float(feature_importance[idx]) for idx in sorted_idx}
        except:
            pass 
        
        if len(asTestParquetFiles) > 0 and not bOperationalMode:
            total_confusion_matrix = np.zeros((2, 2))
            for idx, sTestFile in enumerate(asTestParquetFiles):            
                wasdi.updateProgressPerc(80 + int((idx / len(asTestParquetFiles)) * 10))
                df_test_map = pd.read_parquet(sTestFile)
                y_actual = df_test_map['FloodedNonFlooded'].values
                y_actual_binary = (y_actual > 0.0).astype(int)
                
                # SYNCED FAILSAFE FOR METRICS 
                bDryDay = False
                if 'RainCum_24hr' in df_test_map.columns and df_test_map['RainCum_24hr'].max() < 10.0: # <--- Upgraded physically-backed threshold
                    bDryDay = True

                if bDryDay:
                    y_pred_chunk_binary = np.zeros(len(df_test_map))
                else:
                    X_test_chunk = df_test_map[asFeatureNames].astype('float64')
                    y_pred_chunk = oModel.predict(X_test_chunk)
                    
                    # LOCAL ZONAL FAILSAFE
                    if 'RainCum_24hr' in df_test_map.columns:
                        afRain24_chunk = df_test_map['RainCum_24hr'].values
                        y_pred_chunk[afRain24_chunk < 10.0] = 0.0 # <--- Upgraded physically-backed threshold
                    
                    y_pred_chunk_binary = (y_pred_chunk >= fThreshold).astype(int)
                    del X_test_chunk
                
                total_confusion_matrix += confusion_matrix(y_actual_binary, y_pred_chunk_binary, labels=[0, 1])
                del df_test_map
            
            # Calculate standard matrix
            TN = int(total_confusion_matrix[0][0])
            FP = int(total_confusion_matrix[0][1])
            FN = int(total_confusion_matrix[1][0])
            TP = int(total_confusion_matrix[1][1])
            
            # Safely calculate advanced metrics (preventing division by zero)
            fPrecision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            fRecall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            fF1_Score = 2 * (fPrecision * fRecall) / (fPrecision + fRecall) if (fPrecision + fRecall) > 0 else 0.0
            
            aoPayload[f"Test Set Metrics (Threshold {fThreshold})"] = {
                "True Negatives": TN,
                "False Positives": FP,
                "False Negatives": FN,
                "True Positives": TP,
                "Precision": round(fPrecision, 4),
                "Recall": round(fRecall, 4),
                "F1_Score": round(fF1_Score, 4)
            }
        
        wasdi.setPayload(aoPayload)
        wasdi.updateProgressPerc(90) 

        # --- 7. EXPORT MAPS TO WASDI ---
        if len(asTestParquetFiles) > 0:
            wasdi.wasdiLog("Transforming Zonal predictions into 2D Heatmaps...")
            oFloodMapTemp = gdal.Open(wasdi.getPath(asFloodMaps[0]))
            iRows, iCols = oFloodMapTemp.RasterYSize, oFloodMapTemp.RasterXSize

            for idx, sTestFile in enumerate(asTestParquetFiles):
                wasdi.updateProgressPerc(90 + int((idx / len(asTestParquetFiles)) * 8))

                df_test_map = pd.read_parquet(sTestFile)
                sUniqueDate = df_test_map['Date'].iloc[0]
                afPredictedFloatFloodMap = np.zeros((iRows, iCols))

                # NO RAIN FAILSAFE
                bDryDay = False
                if 'RainCum_24hr' in df_test_map.columns:
                    fMaxRain24h = df_test_map['RainCum_24hr'].max()
                    if fMaxRain24h < 10.0: # <--- Upgraded physically-backed threshold
                        wasdi.wasdiLog(f"FAILSAFE TRIGGERED: Max 24h rain is only {fMaxRain24h:.2f}mm. Bypassing AI.")
                        bDryDay = True

                if not bDryDay:
                    X_test_map = df_test_map[asFeatureNames].astype('float64')
                    # Forces the raw XGBoost output to strictly stay between 0.0 and 1.0
                    y_pred_map = np.clip(oModel.predict(X_test_map), 0.0, 1.0)

                    # LOCAL ZONAL FAILSAFE
                    if 'RainCum_24hr' in df_test_map.columns:
                        afRain24 = df_test_map['RainCum_24hr'].values
                        # Overwrite XGBoost: Force dry zones to strictly 0% risk
                        y_pred_map[afRain24 < 10.0] = 0.0 # <--- Upgraded physically-backed threshold

                    afZoneX = df_test_map['Zone_X'].values.astype(int)
                    afZoneY = df_test_map['Zone_Y'].values.astype(int)

                    # VECTORIZED ZONAL FILLING 
                    for iIndex in range(len(y_pred_map)):                         
                        # Reconstruct the starting pixel coordinates from the zone indices
                        iStartRow = afZoneY[iIndex] * iZoneSize
                        iEndRow = min(iStartRow + iZoneSize, iRows)
                        iStartCol = afZoneX[iIndex] * iZoneSize
                        iEndCol = min(iStartCol + iZoneSize, iCols)
                        
                        # Populate the entire block with the predicted probability
                        afPredictedFloatFloodMap[iStartRow:iEndRow, iStartCol:iEndCol] = y_pred_map[iIndex]

                    del X_test_map

                sPredictedFloatFloodMap = f"{sBasenamefloodmap}_{sUniqueDate}_ZonalFloatFlood.tif"
                oDriver = gdal.GetDriverByName("GTiff")
                oOutputMap = oDriver.Create(wasdi.getPath(sPredictedFloatFloodMap), iCols, iRows, 1, gdal.GDT_Float32, ['COMPRESS=LZW', 'BIGTIFF=YES'])
                oOutputMap.SetGeoTransform(oFloodMapTemp.GetGeoTransform())
                oOutputMap.SetProjection(oFloodMapTemp.GetProjection())
                oOutputMap.GetRasterBand(1).WriteArray(afPredictedFloatFloodMap)
                oOutputMap.FlushCache()
                wasdi.addFileToWASDI(sPredictedFloatFloodMap, "FloodForecastZonal_0-1")
                del df_test_map

    except Exception as oE:
        wasdi.wasdiLog("Error: " + str(oE))
        wasdi.setPayload(aoPayload)
        wasdi.updateStatus('ERROR')
        return

    wasdi.setPayload(aoPayload)
    wasdi.wasdiLog('Done! Processing Completed Successfully.')
    wasdi.updateProgressPerc(100)
    wasdi.updateStatus('DONE', 100)


if __name__ == '__main__':
    wasdi.init('./config.json')
    run()