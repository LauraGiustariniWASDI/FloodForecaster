import wasdi
import re
import os
from datetime import datetime
from osgeo import gdal, osr
import numpy as np
import pandas as pd

from whitebox.whitebox_tools import WhiteboxTools
from pysheds.grid import Grid
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import glob


def run():

    wasdi.wasdiLog("START: Flood Forecaster v.2.1.0")
    aoPayload = {}

    try:
        sBasenamefloodmap = wasdi.getParameter("BASENAME_FLOODMAP", "PWThies") 
        sSuffixfloodmap = wasdi.getParameter("SUFFIX_FLOODMAP", "_flood.tif") 
        sBasenameimerg = wasdi.getParameter("BASENAME_IMERG", "Thies_Cumulative_") 
        
        # HYDROLOGY PARAMETERS
        sDem = wasdi.getParameter("DEM", "Thies_DEM30m.tif")
        bFillDem = wasdi.getParameter("FILL_DEM", True)
        bComputeTwi = wasdi.getParameter("COMPUTE_TWI", False)
        sTwiMap = wasdi.getParameter("TWI_MAP", "")
        bComputeHand = wasdi.getParameter("COMPUTE_HAND", False)
        sHandMap = wasdi.getParameter("HAND_MAP", "")
        iMinAccValueRaw = wasdi.getParameter("MIN_ACC_VALUE_HAND", 200)
        iMinAccValue = int(iMinAccValueRaw) if iMinAccValueRaw else 200
        sLulcMap = wasdi.getParameter("LULC_MAP", "Thies_LULC.tif")
        bUseLulc = (sLulcMap != "")
        
        # OPERATIONAL, BATCHING & ML PARAMETERS
        bOperationalMode = wasdi.getParameter("OPERATIONAL", False)
        bReprocessAll = wasdi.getParameter("REPROCESS_ALL", False)
        sTestDate = str(wasdi.getParameter("TEST_DATE", "")).strip() 
        sForecastDateTime = str(wasdi.getParameter("FORECAST_DATETIME", "")).strip()
        
        # --- OPERATIONAL FRESNESS SAFEGUARD ---
        if bOperationalMode:
            bReprocessAll = True
        # --------------------------------------

        # BATCHING PARAMETERS
        iStartMapRaw = wasdi.getParameter("START_MAP_INDEX", 1)
        iEndMapRaw = wasdi.getParameter("END_MAP_INDEX", "")
        
        # Convert user-friendly 1-based index to Python's 0-based index
        iStartMap = max(0, int(iStartMapRaw) - 1) if iStartMapRaw not in ["", None] else 0
        iEndMap = int(iEndMapRaw) if iEndMapRaw not in ["", None] else None
        
        fThresholdRaw = wasdi.getParameter("THRESHOLD", None)
        fThreshold = float(fThresholdRaw) if fThresholdRaw not in ["", None] else 0.5
        sListMapsWithFloodTxt = wasdi.getParameter("LIST_MAPS_WITH_FLOOD", "") 
        sAlgorithmRaw = wasdi.getParameter("ALGORITHM", "XGBoost")
        sAlgorithm = str(sAlgorithmRaw).strip().lower() if sAlgorithmRaw else "xgboost"
        sTechnique = str(wasdi.getParameter("TECHNIQUE", "regression")).strip().lower()
        sModelBaselineJoblib = wasdi.getParameter("BASELINE_MODEL", "")
        bSaveBaselineModel = wasdi.getParameter("SAVE_BASELINE_MODEL", False)

        aoPayload["INPUTS"] = wasdi.getParametersDict()
        wasdi.setPayload(aoPayload)
        wasdi.wasdiLog("All input parameters read")
        wasdi.updateProgressPerc(5) # Parameters processed

        # --- 1. COMPUTE HYDROLOGICAL VARIABLES ---
        if bComputeTwi:
            wasdi.wasdiLog("Computing Topographic Wetness Index (TWI)...")
            wbt = WhiteboxTools()
            
            if bFillDem:
                wasdi.wasdiLog("Filling depressions/sinks in DEM")
                sFilledDem = sDem.replace(".tif", "_filled.tif")
                wbt.fill_depressions(dem=wasdi.getPath(sDem), output=wasdi.getPath(sFilledDem))
            else:
                wasdi.wasdiLog("Using input DEM, without filling depressions/sinks")
                sFilledDem = sDem

            sSlope = sDem.replace(".tif", "_slope.tif")
            wbt.slope(dem=wasdi.getPath(sFilledDem), output=wasdi.getPath(sSlope), units="degrees")

            sSpecContrArea = sDem.replace(".tif", "_sca.tif")
            wbt.d8_flow_accumulation(wasdi.getPath(sFilledDem), wasdi.getPath(sSpecContrArea), out_type="specific contributing area")

            sTWI = sDem.replace(".tif", "_TWI.tif")
            wbt.wetness_index(sca=wasdi.getPath(sSpecContrArea), slope=wasdi.getPath(sSlope), output=wasdi.getPath(sTWI))
            
            wasdi.addFileToWASDI(sTWI)
            sTwiMap = sTWI 
            wasdi.wasdiLog(f"TWI Computation complete and saved to workspace as {sTwiMap}.")
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
            wasdi.wasdiLog(f"HAND Computation complete and saved to workspace as {sHandMap}.")
        elif sHandMap != "":
            wasdi.wasdiLog(f"Using pre-computed HAND map: {sHandMap}")

        wasdi.updateProgressPerc(10) # Hydrology Phase complete

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
            
            # Sort maps to ensure batch slices are perfectly deterministic
            asAllAvailableMaps.sort()
            asFloodMaps = asAllAvailableMaps[iStartMap:iEndMap]
            
            iEndLog = iEndMap if iEndMap else len(asAllAvailableMaps)
            wasdi.wasdiLog(f"Batch Filter Applied: Selected {len(asFloodMaps)} flood maps (from index {iStartMap+1} to {iEndLog}) out of {len(asAllAvailableMaps)} total.")

        else:
            wasdi.wasdiLog(f"Reading flood maps from the file: {sListMapsWithFloodTxt}")
            with open(wasdi.getPath(sListMapsWithFloodTxt), "r") as f:
                asAllAvailableMaps = f.read().splitlines()
                asAllAvailableMaps.sort() # CRITICAL: Ensure consistent ordering
                asFloodMaps = asAllAvailableMaps[iStartMap:iEndMap]
                
            iEndLog = iEndMap if iEndMap else len(asAllAvailableMaps)
            wasdi.wasdiLog(f"Batch Filter Applied: Selected {len(asFloodMaps)} flood maps (from index {iStartMap+1} to {iEndLog}) out of {len(asAllAvailableMaps)} total.")

        # --- 2.5 APPLY TEST/FORECAST & OPERATIONAL FILTER ---
        sGFSTargetTime = ""
        bUseGFS = False
        asFutureRainMaps = []
        asRainCumulVals = ["1hr", "3hr", "6hr", "12hr", "24hr"]

        if bOperationalMode:
            if sForecastDateTime == "":
                raise Exception("CRITICAL ERROR: OPERATIONAL mode is True, but FORECAST_DATETIME is empty.")
            wasdi.wasdiLog(f"OPERATIONAL FORECAST MODE ACTIVE for target: {sForecastDateTime}")
            bUseGFS = True
            sGFSTargetTime = sForecastDateTime
            
            # A. Safely parse the target datetime to an exact numeric string (e.g., "202604021200")
            try:
                dtTarget = datetime.strptime(sGFSTargetTime, '%Y-%m-%d %H:%M')
                sTargetDateDigits = dtTarget.strftime("%Y%m%d")
                sTargetTimeDigits = dtTarget.strftime("%H%M")
            except ValueError:
                # Fallback if time isn't provided perfectly
                sTargetDateDigits = sGFSTargetTime.replace("-", "").replace(" ", "").replace(":", "")
                sTargetTimeDigits = ""

            sFullTargetDigits = sTargetDateDigits + sTargetTimeDigits
            
            # B. Search workspace for files matching the BaseName AND the exact Target Datetime
            asPotentialMaps = []
            for sFile in asCurrentFiles:
                if sBasenameimerg in sFile:
                    # Strip symbols from filename to prevent formatting mismatches
                    sFileDigits = "".join([c for c in sFile if c.isdigit()])
                    
                    # Ensure the exact date and time sequence exists in the filename
                    if sFullTargetDigits in sFileDigits or (sTargetTimeDigits == "" and sTargetDateDigits in sFileDigits):
                        asPotentialMaps.append(sFile)
            
            # C. Verify all 5 required cumulative periods exist for this exact time
            for sPeriod in asRainCumulVals:
                for sFile in asPotentialMaps:
                    if sPeriod in sFile and sFile not in asFutureRainMaps:
                        asFutureRainMaps.append(sFile)
                        break # Found the map for this period

            # D. Trigger Cumulator if ANY of the 5 periods are missing
            if len(asFutureRainMaps) < 5:
                wasdi.wasdiLog(f"Missing required GFS/IMERG cumulates for exact datetime {sGFSTargetTime}. Triggering Cumulator App...")
                
                # Extract BBOX from DEM for the app
                oDemData = gdal.Open(wasdi.getPath(sDem))
                fUlx, fXres, fXskew, fUly, fYskew, fYres = oDemData.GetGeoTransform()
                iRows, iCols = oDemData.RasterYSize, oDemData.RasterXSize
                fLrx = fUlx + (iCols * fXres)
                fLry = fUly + (iRows * fYres)
                oDemData = None
                
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
                
                sGFSCumulatorId = wasdi.executeProcessor("gfs_precipitation_cumulator", aoGFSDict)
                wasdi.wasdiLog(f"Waiting for Cumulator App (Process ID: {sGFSCumulatorId})...")
                wasdi.waitProcesses([sGFSCumulatorId])
                
                # Retrieve the newly generated files
                aoGFSPayload = wasdi.getProcessorPayloadAsJson(sGFSCumulatorId)
                asFutureRainMaps = aoGFSPayload.get("OUTPUTS", [])
                wasdi.wasdiLog(f"Cumulator finished. Retrieved {len(asFutureRainMaps)} forecast maps.")
            else:
                wasdi.wasdiLog(f"Found all 5 required cumulative maps in workspace for {sGFSTargetTime}. Skipping Cumulator app.")
            
            if len(asFutureRainMaps) == 0:
                raise Exception("CRITICAL ERROR: Failed to retrieve forecasting data.")
            
            # E. Create a dummy list to force Phase 3 to run ONCE using the first available map as a spatial template
            if len(asAllAvailableMaps) == 0:
                raise Exception("CRITICAL ERROR: No historical map found to use as a spatial grid template.")
            asFloodMaps = [asAllAvailableMaps[0]] 

        elif sTestDate != "":
            # Standard Historical Testing Mode
            asFloodMaps = [f for f in asFloodMaps if sTestDate in f]
            if len(asFloodMaps) == 0:
                raise Exception(f"CRITICAL ERROR: No input flood map found matching TEST_DATE '{sTestDate}'.")
            wasdi.wasdiLog(f"HISTORICAL TEST MODE: Filtered dataset to {len(asFloodMaps)} map(s).")
            
            if sModelBaselineJoblib == "":
                raise Exception("CRITICAL ERROR: You specified a TEST_DATE but did not provide a BASELINE_MODEL to test it with.")

        asIMERGMaps = [f for f in asCurrentFiles if sBasenameimerg in f]
        aoPayload["Total Flood Maps Processing"] = len(asFloodMaps) if not bOperationalMode else 1
        wasdi.setPayload(aoPayload)
        wasdi.updateProgressPerc(20)

        # --- 3. BUILD DATAFRAME (ITERATIVE PARQUET METHOD) ---
        if len(asFloodMaps) > 0:
            wasdi.wasdiLog("Building Tabular Dataset iteratively on disk via Parquet...")
        
        bUseTwi = (sTwiMap != ""); bUseHand = (sHandMap != "")
        asRainCumulVals = ["1hr", "3hr", "6hr", "12hr", "24hr"]

        for iCounter, sFloodMap in enumerate(asFloodMaps):
            wasdi.wasdiLog(f"Processing Spatial Grid: {sFloodMap}")

            # MISSING FILE DETECTOR
            for sRequiredFile in [sFloodMap, sDem, sTwiMap, sHandMap]:
                if not os.path.exists(wasdi.getPath(sRequiredFile)):
                    raise Exception(f"CRITICAL ERROR: The file '{sRequiredFile}' is missing from the workspace!")
            
            iCurrentProgress = 20 + int(((iCounter) / len(asFloodMaps)) * 30)
            wasdi.updateProgressPerc(iCurrentProgress)
            
            # If operational, force the date to the future forecast date so the Parquet saves correctly
            sMapDateRaw = re.search(r'\d{4}-\d{2}-\d{2}', sFloodMap).group()
            sDate = sGFSTargetTime.split(" ")[0] if bOperationalMode else sMapDateRaw
            
            sParquetName = wasdi.getPath(f"temp_map_data_{sDate}.parquet")
            
            if os.path.exists(sParquetName) and not bReprocessAll:
                wasdi.wasdiLog(f"Parquet already exists for {sDate}. Skipping warp phase.")
                continue
            elif os.path.exists(sParquetName) and bReprocessAll:
                wasdi.wasdiLog(f"REPROCESS_ALL active: Overwriting existing Parquet for {sDate}.")

            oFloodMap = gdal.Open(wasdi.getPath(sFloodMap))
            # CORRUPTED/EMPTY FILE FAILSAFE
            if oFloodMap is None:
                wasdi.wasdiLog(f"WARNING: GDAL cannot read {sFloodMap}. It is likely 0-bytes or corrupted. Skipping map.")
                continue

            fUlx, fXres, fXskew, fUly, fYskew, fYres = oFloodMap.GetGeoTransform()
            iRows, iCols = oFloodMap.RasterYSize, oFloodMap.RasterXSize
            
            afColGrid, afRowGrid = np.meshgrid(np.arange(iCols) + 0.5, np.arange(iRows) + 0.5)
            afLon1D = (fUlx + afColGrid * fXres + afRowGrid * fXskew).ravel()
            afLat1D = (fUly + afColGrid * fYskew + afRowGrid * fYres).ravel()
            afColGrid0, afRowGrid0 = np.meshgrid(np.arange(iCols), np.arange(iRows))

            afFloodMap1D = np.array(oFloodMap.GetRasterBand(1).ReadAsArray()).ravel()
            valid_mask = (afFloodMap1D != 0) & (afFloodMap1D != 2)
            if np.sum(valid_mask) == 0: continue
                
            dict_valid_data = {
                'FloodedNonFlooded': afFloodMap1D[valid_mask],
                'Lat': afLat1D[valid_mask], 'Lon': afLon1D[valid_mask],
                'PosX': afColGrid0.ravel()[valid_mask], 'PosY': afRowGrid0.ravel()[valid_mask]
            }
            del afColGrid, afRowGrid, afLon1D, afLat1D, afColGrid0, afRowGrid0
            
            afBbox = [fUlx, fUly + (iRows * fYres), fUlx + (iCols * fXres), fUly]
            sDestProj = oFloodMap.GetProjection()
            
            # By adding srcSRS=sDestProj, we force GDAL to warp the 15m maps 
            aoWarpOptions = gdal.WarpOptions(
                srcSRS=sDestProj, 
                dstSRS=sDestProj, 
                xRes=fXres, 
                yRes=abs(fYres), 
                outputBounds=afBbox, 
                format="MEM"
            )

            # STRICT DEM CHECK
            oDem = gdal.Open(wasdi.getPath(sDem))
            oWarpedDem = gdal.Warp("", oDem, options=aoWarpOptions)
            if oWarpedDem is None:
                raise Exception(f"CRITICAL ERROR: gdal.Warp failed on DEM: {sDem}. Check metadata.")
            afWarpedDem1D = np.array(oWarpedDem.GetRasterBand(1).ReadAsArray()).ravel()
            dict_valid_data['DEM'] = afWarpedDem1D[valid_mask]
            oDem = None; oWarpedDem = None 

            if bUseTwi: 
                oTWI = gdal.Open(wasdi.getPath(sTwiMap))
                oWarpedTWI = gdal.Warp("", oTWI, options=aoWarpOptions)
                if oWarpedTWI is None:
                    raise Exception(f"CRITICAL ERROR: gdal.Warp failed on TWI: {sTwiMap}. Check metadata.")
                dict_valid_data['TWI'] = np.array(oWarpedTWI.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
                oTWI = None; oWarpedTWI = None
                
            if bUseHand: 
                oHAND = gdal.Open(wasdi.getPath(sHandMap))
                oWarpedHAND = gdal.Warp("", oHAND, options=aoWarpOptions)
                if oWarpedHAND is None:
                    raise Exception(f"CRITICAL ERROR: gdal.Warp failed on HAND: {sHandMap}. Check metadata.")
                dict_valid_data['HAND'] = np.array(oWarpedHAND.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
                oHAND = None; oWarpedHAND = None

            if bUseLulc: 
                oLULC = gdal.Open(wasdi.getPath(sLulcMap))
                if oLULC is None:
                    raise Exception(f"CRITICAL ERROR: Failed to open LULC: {sLulcMap}.")
                
                # Categorical data MUST use Nearest Neighbour
                aoWarpOptionsLULC = gdal.WarpOptions(
                    srcSRS=sDestProj, 
                    dstSRS=sDestProj, 
                    xRes=fXres, 
                    yRes=abs(fYres), 
                    outputBounds=afBbox, 
                    format="MEM",
                    resampleAlg=gdal.GRA_NearestNeighbour 
                )
                
                oWarpedLULC = gdal.Warp("", oLULC, options=aoWarpOptionsLULC)
                if oWarpedLULC is None:
                    raise Exception(f"CRITICAL ERROR: gdal.Warp failed on LULC: {sLulcMap}.")
                
                # Extract the raw 10, 20, 50, etc. values    
                afRawLULC = np.array(oWarpedLULC.GetRasterBand(1).ReadAsArray()).ravel()[valid_mask]
                
                # Perform One-Hot Encoding mapping ESA classes to Binary (0 or 1)
                dict_valid_data['LULC_Trees'] = (afRawLULC == 10).astype(int)
                dict_valid_data['LULC_ShrubGrass'] = np.isin(afRawLULC, [20, 30]).astype(int)
                dict_valid_data['LULC_Crop'] = (afRawLULC == 40).astype(int)
                dict_valid_data['LULC_BuiltUp'] = (afRawLULC == 50).astype(int)
                dict_valid_data['LULC_Bare'] = (afRawLULC == 60).astype(int)
                dict_valid_data['LULC_WaterWetland'] = np.isin(afRawLULC, [80, 90, 95]).astype(int)
                
                oLULC = None; oWarpedLULC = None; del afRawLULC

            dict_valid_data['Date'] = [sDate] * np.sum(valid_mask)
            
            # --- METEOROLOGICAL DATA INGESTION ---
            if bOperationalMode:
                # Use the future GFS maps we just generated
                for sRainMap in asFutureRainMaps:
                    for sCumulVal in asRainCumulVals:
                        if sCumulVal in sRainMap:
                            oRainData = gdal.Open(wasdi.getPath(sRainMap))
                            oWarpedRain = gdal.Warp("", oRainData, options=aoWarpOptions)
                            afRainMap1D = np.array(oWarpedRain.GetRasterBand(1).ReadAsArray()).ravel()
                            dict_valid_data["RainCum_" + sCumulVal] = afRainMap1D[valid_mask]
                            oRainData = None; oWarpedRain = None
            else:
                # Use standard historical IMERG maps
                sIMERGDate = datetime.strptime(sDate, '%Y-%m-%d').strftime("%Y%m%d")
                asDateIMERGMaps = [m for m in asIMERGMaps if f"-{sIMERGDate}" in m]
                
                for sIMERGMap in asDateIMERGMaps:
                    for sCumulVal in asRainCumulVals:
                        if sCumulVal in sIMERGMap:
                            oIMERG = gdal.Open(wasdi.getPath(sIMERGMap))
                            if oIMERG is None:
                                raise Exception(f"CRITICAL ERROR: Failed to open IMERG map: {sIMERGMap}")
                                
                            oWarpedIMERG = gdal.Warp("", oIMERG, options=aoWarpOptions)
                            if oWarpedIMERG is None:
                                raise Exception(f"CRITICAL ERROR: gdal.Warp failed on IMERG map: {sIMERGMap}. Check map boundaries and projection.")
                                
                            afIMERGMap1D = np.array(oWarpedIMERG.GetRasterBand(1).ReadAsArray()).ravel()
                            dict_valid_data["RainCum_" + sCumulVal] = afIMERGMap1D[valid_mask]
                            oIMERG = None; oWarpedIMERG = None

            dfFloodInLoop = pd.DataFrame(dict_valid_data)
            
            if bUseTwi: dfFloodInLoop['TWI'] = dfFloodInLoop['TWI'].fillna(0)
            if bUseHand: dfFloodInLoop['HAND'] = dfFloodInLoop['HAND'].fillna(9999)

            # NEW INTERACTION FEATURES
            if 'RainCum_24hr' in dfFloodInLoop.columns:
                if bUseHand:
                    # Flood Index: High Rain / Low Elevation = Danger
                    dfFloodInLoop['Interaction_Rain24_HAND'] = dfFloodInLoop['RainCum_24hr'] / (dfFloodInLoop['HAND'] + 1.0)
                if bUseTwi:
                    # Saturation: High Rain * High Wetness Potential = Danger
                    dfFloodInLoop['Interaction_Rain24_TWI'] = dfFloodInLoop['RainCum_24hr'] * dfFloodInLoop['TWI']

            asColsToCheck = [col for col in dfFloodInLoop.columns if col != 'FloodedNonFlooded']
            dfFloodInLoop = dfFloodInLoop.dropna(subset=asColsToCheck)
            
            dfFloodInLoop.loc[dfFloodInLoop["FloodedNonFlooded"] == 1, "FloodedNonFlooded"] = 0
            dfFloodInLoop.loc[dfFloodInLoop["FloodedNonFlooded"] == 3, "FloodedNonFlooded"] = 1

            dfFloodInLoop.to_parquet(sParquetName, engine='fastparquet', compression='snappy')
            del dict_valid_data, dfFloodInLoop, afWarpedDem1D, afFloodMap1D, valid_mask

        wasdi.updateProgressPerc(50) # Data Building Phase complete

        # --- 4. DATA ROUTING (TRAINING vs TESTING) ---
        asAllParquetFiles = []
        for sMap in asFloodMaps:
            if bOperationalMode:
                # Look for the future GFS forecast date Parquet we just created
                sMapDate = sForecastDateTime.split(" ")[0]
            else:
                # Look for the historical date Parquet
                sMapDate = re.search(r'\d{4}-\d{2}-\d{2}', sMap).group()
                
            sExpectedPath = wasdi.getPath(f"temp_map_data_{sMapDate}.parquet")
            
            if os.path.exists(sExpectedPath):
                asAllParquetFiles.append(sExpectedPath)
                
        asTrainParquetFiles = []
        asTestParquetFiles = []

        if sTestDate == "" and not bOperationalMode:
            wasdi.wasdiLog(f"Mode: TRAINING ONLY. Routing {len(asAllParquetFiles)} maps to the training batch loop.")
            asTrainParquetFiles = asAllParquetFiles
        else:
            sTargetLog = sForecastDateTime if bOperationalMode else sTestDate
            wasdi.wasdiLog(f"Mode: INFERENCE ONLY. Routing map {sTargetLog} to the evaluation block.")
            asTestParquetFiles = asAllParquetFiles

        wasdi.updateProgressPerc(55) # Routing complete
        
        # --- 5. MODEL INITIALIZATION & TRAINING ---
        bIsXGBoost = sAlgorithm not in ["random forest", "randomforest", "rf"]
        BATCH_SIZE = 30 
        
        # Deterministically build Feature Names to prevent Scikit-Learn mismatches
        asFeatureNames = ['DEM']
        if bUseTwi: 
            asFeatureNames.append('TWI')
            asFeatureNames.append('Interaction_Rain24_TWI')
        if bUseHand: 
            asFeatureNames.append('HAND')
            asFeatureNames.append('Interaction_Rain24_HAND')
        if bUseLulc: # THE ONE-HOT ENCODED LULC FEATURES
            asFeatureNames.extend(['LULC_Trees', 'LULC_ShrubGrass', 'LULC_Crop', 'LULC_BuiltUp', 'LULC_Bare', 'LULC_WaterWetland'])
        asFeatureNames.extend(["RainCum_1hr", "RainCum_3hr", "RainCum_6hr", "RainCum_12hr", "RainCum_24hr"])
        
        if sModelBaselineJoblib == "":
            wasdi.wasdiLog(f"Initializing new {sAlgorithm} model...")
            if not bIsXGBoost:
                iTotalBatches = max(1, (len(asTrainParquetFiles) + BATCH_SIZE - 1) // BATCH_SIZE)
                iTreesPerBatch = max(5, int(200 / iTotalBatches)) 
                params = {"n_estimators": 0, "max_features": "sqrt", "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 4, "bootstrap": False, "warm_start": True, "random_state": 42, "n_jobs": -1}
                ModelClass = RandomForestRegressor if sTechnique == "regression" else RandomForestClassifier
                oModel = ModelClass(**params)
            else:
                params = {"n_estimators": 50, "max_depth": 7, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1}
                ModelClass = XGBRegressor if sTechnique == "regression" else XGBClassifier
                oModel = ModelClass(**params)
        else:
            wasdi.wasdiLog(f"Loading pre-trained model {sModelBaselineJoblib}...")
            oModel = joblib.load(wasdi.getPath(sModelBaselineJoblib))
            if not bIsXGBoost: 
                iTreesPerBatch = 10 

        aoPayload["Algorithm Used"] = "XGBoost" if bIsXGBoost else "Random Forest"
        aoPayload["Technique Used"] = sTechnique.capitalize()
        wasdi.setPayload(aoPayload)

        # Execute Incremental Training ONLY if we have training files
        if len(asTrainParquetFiles) > 0:
            columns_to_drop = ['FloodedNonFlooded', 'Date', 'PosX', 'PosY', 'Lat', 'Lon']
            
            iTotalTrainingBatches = (len(asTrainParquetFiles) + BATCH_SIZE - 1) // BATCH_SIZE
            iBatchCounter = 0
            
            for i in range(0, len(asTrainParquetFiles), BATCH_SIZE):
                
                # Dynamic Progress: Maps to 55% -> 80%
                iCurrentProgress = 55 + int((iBatchCounter / iTotalTrainingBatches) * 25)
                wasdi.updateProgressPerc(iCurrentProgress)
                iBatchCounter += 1
                
                asBatchFiles = asTrainParquetFiles[i:i+BATCH_SIZE]
                wasdi.wasdiLog(f"Training Phase: Loading Batch {int(i/BATCH_SIZE)+1} containing {len(asBatchFiles)} maps...")
                
                asDataFramesToConcat = []
                for sFile in asBatchFiles:
                    df_Map = pd.read_parquet(sFile)                    
                    df_flooded = df_Map[df_Map['FloodedNonFlooded'] == 1]
                    df_dry = df_Map[df_Map['FloodedNonFlooded'] == 0]
                    
                    if len(df_flooded) > 0:
                        # Normal Map: Sample dry pixels at a 10:1 ratio
                        iTargetDryCount = len(df_flooded) * 10
                        if len(df_dry) > iTargetDryCount:
                            df_dry = df_dry.sample(n=iTargetDryCount, random_state=42)
                    else:
                        # DRY DAY MAP: The user wants to teach the AI the baseline.
                        # We grab exactly 10,000 random dry pixels so the AI learns the "Off Switch" without destroying the 10:1 balance of the overall batch.
                        if len(df_dry) > 10000:
                            df_dry = df_dry.sample(n=10000, random_state=42)
                        
                    asDataFramesToConcat.append(df_flooded)
                    asDataFramesToConcat.append(df_dry)
                    del df_Map
                    
                df_batch = pd.concat(asDataFramesToConcat, ignore_index=True).sample(frac=1, random_state=42)
                del asDataFramesToConcat
                
                X_base = df_batch.drop(columns_to_drop, axis=1, errors='ignore').astype('float64')
                y_base = df_batch[['FloodedNonFlooded']]
                X_base = X_base[asFeatureNames] # Enforce precise column order
                
                # DATA AUGMENTATION (JITTERING)
                # Identify columns that are purely meteorological or meteorological-driven
                asRainAndInteractionCols = [c for c in X_base.columns if "RainCum" in c or "Interaction" in c]
                
                # Create a scenario where it rained 20% LESS
                X_down = X_base.copy()
                X_down[asRainAndInteractionCols] = X_down[asRainAndInteractionCols] * 0.8
                
                # Create a scenario where it rained 20% MORE
                X_up = X_base.copy()
                X_up[asRainAndInteractionCols] = X_up[asRainAndInteractionCols] * 1.2
                
                # Stack them all together to triple the training size
                X_batch_final = pd.concat([X_base, X_down, X_up], ignore_index=True)
                y_batch_final = pd.concat([y_base, y_base, y_base], ignore_index=True)
                
                # Shuffle the new massive dataset so the model doesn't memorize chunks
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
                
                wasdi.wasdiLog(f"Batch {int(i/BATCH_SIZE)+1} trained on {len(X_batch_final):,} Augmented pixels (Jitter +/- 20%). RAM cleared.")
                del df_batch, X_base, y_base, X_down, X_up, X_batch_final, y_batch_final

            if bSaveBaselineModel:
                sOutModelName = f"{sBasenamefloodmap}_baseline_model.joblib"
                joblib.dump(oModel, wasdi.getPath(sOutModelName))
                wasdi.addFileToWASDI(sOutModelName)
                wasdi.wasdiLog(f"Model successfully saved to Workspace as {sOutModelName}")
        else:
            wasdi.wasdiLog("Training skipped (Test Mode Active).")

        wasdi.updateProgressPerc(80) # Training Phase complete

        # --- 6. ITERATIVE EVALUATION FOR PAYLOAD METRICS ---
        try:
            feature_importance = oModel.feature_importances_
            sorted_idx = np.argsort(feature_importance)[::-1]
            aoPayload["Feature Importance"] = {asFeatureNames[idx]: float(feature_importance[idx]) for idx in sorted_idx}
        except:
            pass 
        
        if len(asTestParquetFiles) > 0:
            if not bOperationalMode:
                wasdi.wasdiLog("Evaluating model iteratively for payload metrics...")
                total_confusion_matrix = np.zeros((2, 2))

                for idx, sTestFile in enumerate(asTestParquetFiles):            
                    # Dynamic Progress: Maps to 80% -> 90%
                    iCurrentProgress = 80 + int((idx / len(asTestParquetFiles)) * 10)
                    wasdi.updateProgressPerc(iCurrentProgress)

                    df_test_map = pd.read_parquet(sTestFile)
                    y_actual = df_test_map['FloodedNonFlooded'].values
                    
                    # SYNCED FAILSAFE FOR METRICS
                    bDryDay = False
                    if 'RainCum_24hr' in df_test_map.columns and df_test_map['RainCum_24hr'].max() < 1.0:
                        bDryDay = True

                    if bDryDay:
                        y_pred_chunk_binary = np.zeros(len(df_test_map))
                    else:
                        X_test_chunk = df_test_map[asFeatureNames].astype('float64')
                        y_pred_chunk = oModel.predict(X_test_chunk)
                        y_pred_chunk_binary = (y_pred_chunk >= fThreshold).astype(int) if sTechnique == "regression" else y_pred_chunk
                        del X_test_chunk 
                    
                    total_confusion_matrix += confusion_matrix(y_actual, y_pred_chunk_binary, labels=[0, 1])
                    del df_test_map
                
                aoPayload[f"Test Set Metrics (Threshold {fThreshold})"] = {
                    "True Negatives (Not-Flooded)": int(total_confusion_matrix[0][0]),
                    "False Positives (Alarms)": int(total_confusion_matrix[0][1]),
                    "False Negatives (Missed)": int(total_confusion_matrix[1][0]),
                    "True Positives (Flooded)": int(total_confusion_matrix[1][1])
                }
                wasdi.wasdiLog("Evaluation complete. Results saved to payload.")
            else:
                wasdi.wasdiLog("Evaluation skipped (Forecast Mode Active).")
        else:
            wasdi.wasdiLog("Evaluation skipped (Training Mode Active).")

        wasdi.setPayload(aoPayload)
        wasdi.updateProgressPerc(90) # Evaluation Phase complete

        # --- 7. EXPORT MAPS TO WASDI ---
        if len(asTestParquetFiles) > 0:
            wasdi.wasdiLog("Transforming predictions into 2D Maps...")
            oFloodMapTemp = gdal.Open(wasdi.getPath(asFloodMaps[0]))
            iRows, iCols = oFloodMapTemp.RasterYSize, oFloodMapTemp.RasterXSize

            for idx, sTestFile in enumerate(asTestParquetFiles):
                iCurrentProgress = 90 + int((idx / len(asTestParquetFiles)) * 8)
                wasdi.updateProgressPerc(iCurrentProgress)

                df_test_map = pd.read_parquet(sTestFile)
                sUniqueDate = df_test_map['Date'].iloc[0]
                
                wasdi.wasdiLog(f"Generating maps for {sUniqueDate}...")
                
                afPredictedFloatFloodMap = np.zeros((iRows, iCols))
                afPredictedBinaryFloodMap = np.zeros((iRows, iCols))
                afPosX = df_test_map['PosX'].values
                afPosY = df_test_map['PosY'].values

                # NO RAIN FAILSAFE
                bDryDay = False
                if 'RainCum_24hr' in df_test_map.columns:
                    fMaxRain24h = df_test_map['RainCum_24hr'].max()
                    if fMaxRain24h < 1.0: # Less than 1mm of cumulative rain
                        wasdi.wasdiLog(f"FAILSAFE TRIGGERED: Max 24h rain is only {fMaxRain24h:.2f}mm. Bypassing AI and forcing dry map.")
                        bDryDay = True

                if bDryDay:
                    # Model bypass: arrays are instantly populated with zeros
                    y_pred_map = np.zeros(len(df_test_map))
                    afBinaryPreds = np.zeros(len(df_test_map))
                else:
                    # Normal AI Prediction
                    X_test_map = df_test_map[asFeatureNames].astype('float64')
                    y_pred_map = oModel.predict(X_test_map)
                    afBinaryPreds = (y_pred_map >= fThreshold).astype(int) if sTechnique == "regression" else y_pred_map

                # Write Float Predictions (Regression)
                if sTechnique == "regression":
                    for iIndex in range(0, len(y_pred_map)):
                        afPredictedFloatFloodMap[int(afPosY[iIndex]), int(afPosX[iIndex])] = y_pred_map[iIndex]

                    sPredictedFloatFloodMap = f"{sBasenamefloodmap}_{sUniqueDate}_PredictedFloatFlood.tif"
                    oDriver = gdal.GetDriverByName("GTiff")
                    oOutputMap = oDriver.Create(wasdi.getPath(sPredictedFloatFloodMap), iCols, iRows, 1, gdal.GDT_Float32, ['COMPRESS=LZW', 'BIGTIFF=YES'])
                    oOutputMap.SetGeoTransform(oFloodMapTemp.GetGeoTransform())
                    oOutputMap.SetProjection(oFloodMapTemp.GetProjection())
                    oOutputMap.GetRasterBand(1).WriteArray(afPredictedFloatFloodMap)
                    oOutputMap.FlushCache()
                    wasdi.addFileToWASDI(sPredictedFloatFloodMap)

                # Write Binary Predictions
                for iIndex in range(0, len(afBinaryPreds)):
                    afPredictedBinaryFloodMap[int(afPosY[iIndex]), int(afPosX[iIndex])] = afBinaryPreds[iIndex]

                sPredictedBinaryFloodMap = f"{sBasenamefloodmap}_{sUniqueDate}_PredictedBinaryFlood.tif"
                oDriver = gdal.GetDriverByName("GTiff")
                oOutputMap = oDriver.Create(wasdi.getPath(sPredictedBinaryFloodMap), iCols, iRows, 1, gdal.GDT_Int32, ['COMPRESS=LZW', 'BIGTIFF=YES'])
                oOutputMap.SetGeoTransform(oFloodMapTemp.GetGeoTransform())
                oOutputMap.SetProjection(oFloodMapTemp.GetProjection())
                oOutputMap.GetRasterBand(1).WriteArray(afPredictedBinaryFloodMap)
                oOutputMap.FlushCache()
                wasdi.addFileToWASDI(sPredictedBinaryFloodMap)
                
                del df_test_map
                if not bDryDay: del X_test_map

        else:
            wasdi.wasdiLog("Map Generation skipped (Training Mode Active).")

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