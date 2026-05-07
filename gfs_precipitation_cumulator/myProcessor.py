import wasdi
from datetime import datetime, timedelta
from osgeo import gdal
import numpy as np
import math
import os

# --- Helper Functions ---

def roundToNearestHour(oInputTime):
    """Rounds the given time down to the nearest hour for GFS matching."""
    return oInputTime.replace(minute=0, second=0, microsecond=0)

def alignBboxToGFSGrid(oBBox):
    """
    Adjusts the bounding box coordinates to align with the GFS 0.25° grid.
    """
    fGFSResolution = 0.25  # degrees

    fNorth = oBBox["northEast"]["lat"]
    fEast = oBBox["northEast"]["lng"]
    fSouth = oBBox["southWest"]["lat"]
    fWest = oBBox["southWest"]["lng"]

    def roundToGrid(value, resolution, direction='down'):
        if direction == 'down':
            return math.floor(value / resolution) * resolution
        else:
            return math.ceil(value / resolution) * resolution

    fNewNorth = min(roundToGrid(fNorth, fGFSResolution, 'up'), 90.0)
    fNewEast = min(roundToGrid(fEast, fGFSResolution, 'up'), 180.0)
    fNewSouth = max(roundToGrid(fSouth, fGFSResolution, 'down'), -90.0)
    fNewWest = max(roundToGrid(fWest, fGFSResolution, 'down'), -180.0)

    oAlignedBBox = {
        "northEast": {"lat": fNewNorth, "lng": fNewEast},
        "southWest": {"lat": fNewSouth, "lng": fNewWest}
    }
    return oAlignedBBox

def filterGFSFilesForPeriod(asAllFiles, oStartTime, oEndTime):
    """
    Filter GFS files that fall within the given time period.
    Format expected: {BaseName}_{ModelDate}_{ModelRun}_{ForcastHour}_{ProductType}_{ProductLevel}_GFS.tif
    """
    asFilteredFiles = []
    
    for sFile in asAllFiles:
        try:
            sBaseName = os.path.basename(sFile)
            asParts = sBaseName.replace(".tif", "").split('_')
            
            # Extract from the end to safely ignore any underscores in the BaseName
            # [-1] = GFS, [-2] = ProductLevel, [-3] = ProductType
            sFcstPart = asParts[-4]  # e.g., f006
            sRunPart  = asParts[-5]  # e.g., 00, 06, 12, 18
            sDatePart = asParts[-6]  # e.g., 20260311

            # 1. Calculate the base time of the model run
            oBaseTime = datetime.strptime(f"{sDatePart}{sRunPart}", "%Y-%m-%d%H")            
            # 2. Extract the forecast hour integer (strip the 'f')
            iFcstHours = int(sFcstPart.replace('f', ''))
            
            # 3. Calculate the actual valid time of the forecast file
            oFileTime = oBaseTime + timedelta(hours=iFcstHours)

            # Check if file is strictly within our time window (excluding start, including end)
            if oStartTime < oFileTime <= oEndTime:
                asFilteredFiles.append(sFile)

        except Exception as e:
            wasdi.wasdiLog(f"Skipping file {sFile} - does not match expected GFS format: {str(e)}")
            continue

    return asFilteredFiles

def convertRasterUnits(sInputFile, sOutputFile, fMultiplier):
    """
    Multiplies the entire raster array by a given float (e.g., 3600 to convert mm/s to mm/hr).
    """
    try:
        sInputPath = wasdi.getPath(sInputFile)
        oDataset = gdal.Open(sInputPath)
        oBand = oDataset.GetRasterBand(1)
        aData = oBand.ReadAsArray().astype(np.float32)
        iNoData = oBand.GetNoDataValue()

        # Apply multiplication, ignoring NoData values
        if iNoData is not None:
            aData = np.where(aData == iNoData, iNoData, aData * fMultiplier)
        else:
            aData = aData * fMultiplier

        sOutputPath = wasdi.getPath(sOutputFile)
        oDriver = gdal.GetDriverByName("GTiff")
        oOutDataset = oDriver.Create(sOutputPath, oDataset.RasterXSize, oDataset.RasterYSize, 1, gdal.GDT_Float32)
        
        oOutDataset.SetGeoTransform(oDataset.GetGeoTransform())
        oOutDataset.SetProjection(oDataset.GetProjection())
        
        oOutBand = oOutDataset.GetRasterBand(1)
        oOutBand.WriteArray(aData)
        
        if iNoData is not None:
            oOutBand.SetNoDataValue(iNoData)
            
        oOutBand.FlushCache()
        oOutDataset = None
        oDataset = None
        
        wasdi.addFileToWASDI(sOutputFile)
        return True
    except Exception as e:
        wasdi.wasdiLog(f"Error converting units for {sInputFile}: {str(e)}")
        return False

# --- End of Helper Functions ---


def run():
    wasdi.wasdiLog('START: GFS Precipitation Cumulator v.1.2.2')

    # Read Parameters
    sBaseName = wasdi.getParameter("BASE_NAME", "CODE")
    sTargetDateTime = wasdi.getParameter("TARGET_DATETIME", None)
    oBBox = wasdi.getParameter('BBOX')
    sPeriods = wasdi.getParameter("CUMULATION_PERIODS", "1hr,3hr,6hr,12hr,24hr")
    bDelete = wasdi.getParameter("DELETE", True)
    sRunType = wasdi.getParameter("RUN_TYPE", "LAST")  # Default to "LAST" for production, can be overridden for testing

    aoPayload = {'INPUT': wasdi.getParametersDict()}
    wasdi.setPayload(aoPayload)
    wasdi.updateProgressPerc(5)

    oAlignedBBox = alignBboxToGFSGrid(oBBox)

    # 1. Determine Target Time and Current Time
    oNowUTC = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    
    if not sTargetDateTime:
        oTargetTime = oNowUTC
    else:
        oTargetTime = datetime.strptime(sTargetDateTime, "%Y-%m-%d %H:%M")

    oTargetTime = roundToNearestHour(oTargetTime)
    wasdi.wasdiLog(f"Cumulation target end time: {oTargetTime}")

    # 2. Dynamically Calculate Required Duration & Execute Fetcher
    
    # Dynamically find the max requested period (e.g., extracts 24 from '24hr')
    aiRequestedHours = [int(p.replace('hr', '').strip()) for p in sPeriods.split(',') if 'hr' in p]
    iMaxAccumulationHours = max(aiRequestedHours) if aiRequestedHours else 24

    oEarliestNeededTime = oTargetTime - timedelta(hours=iMaxAccumulationHours)
    wasdi.wasdiLog(f"Earliest data hour required for accumulations: {oEarliestNeededTime}")

    # To guarantee the GFS run covers oEarliestNeededTime, we must ensure the initialization date is AT LEAST the day of oEarliestNeededTime, or the day before if it's too early.    
    # We step back 12 hours from the earliest needed time to find a safe search date.
    # If we need data for 04:00 today, searching "yesterday" guarantees the '18' run  from yesterday will cover it perfectly.
    oSafeInitializationTime = oEarliestNeededTime - timedelta(hours=12)
    # --- Prevent future dates before calculating duration ---
    oTodayUTC = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    oSafeMidnight = oSafeInitializationTime.replace(hour=0, minute=0, second=0, microsecond=0)

    # If the safe date is tomorrow or later, force it to today so the math aligns with the fetcher
    if oSafeMidnight > oTodayUTC:
        wasdi.wasdiLog(f"Safe initialization date is in the future. Clamping back to today's run.")
        oSafeMidnight = oTodayUTC
        oSafeInitializationTime = oTodayUTC
    # -----------------------------------------------------------------

    sSearchDate = oSafeInitializationTime.strftime("%Y-%m-%d")
    
    # Now, calculate how many hours out we need to fetch from that safe date's midnight
    iHoursToTarget = math.ceil((oTargetTime - oSafeMidnight).total_seconds() / 3600.0)

    if iHoursToTarget <= 24:
        sDynamicDuration = "24hr"
    elif iHoursToTarget <= 48:
        sDynamicDuration = "48hr"
    elif iHoursToTarget <= 72:
        sDynamicDuration = "72hr" 
    else:
        # Note: If target is >72 hours away, you will need a 96hr+ parameter in your fetcher!
        wasdi.wasdiLog(f"WARNING: Target is {iHoursToTarget} hours out. Capping fetch duration at 72hr.")
        sDynamicDuration = "72hr"

    wasdi.wasdiLog(f"Triggering GFS Fetcher for safe initialization date {sSearchDate}, fetching out to {sDynamicDuration}.")
    
    aoFetcherParams = {
        "BASE_NAME": sBaseName,
        "SEARCH_DATE": sSearchDate, 
        "DURATION": sDynamicDuration, 
        "RUN_TYPE": sRunType,
        "BBOX": oAlignedBBox,
        "DELETE": True,
        "FORCE_ALL_NEW_FILES": True
    }
    
    # Trigger the underlying gfs_fetcher app
    sFetcherProcessId = wasdi.executeProcessor("gfs_fetcher", aoFetcherParams)
    wasdi.wasdiLog(f"Waiting for GFS Fetcher (Process ID: {sFetcherProcessId})...")
    
    sFetchStatus = wasdi.waitProcesses([sFetcherProcessId])
    wasdi.wasdiLog(f"GFS Fetcher finished with status: {sFetchStatus}")
    wasdi.updateProgressPerc(40)

    # 3. Retrieve resulting GFS files directly from the fetcher's payload
    wasdi.wasdiLog("Retrieving GFS files directly from the GFS Fetcher's payload...")
    try:
        aoExternalAppPayload = wasdi.getProcessorPayloadAsJson(sFetcherProcessId)
        asGFSHourlyFiles = aoExternalAppPayload.get("OUTPUTS", [])
        wasdi.wasdiLog(f"Successfully retrieved {len(asGFSHourlyFiles)} files from fetcher payload.")
    except Exception as e:
        wasdi.wasdiLog(f"ERROR: Could not retrieve payload from fetcher: {str(e)}")
        asGFSHourlyFiles = []

    # Abort gracefully if the fetcher failed to provide data ---
    if not asGFSHourlyFiles:
        wasdi.wasdiLog("ERROR: No hourly GFS files were found in the workspace. Cannot proceed with cumulation.")
        aoPayload["STATUS"] = "FAILED - No source data available from GFS Fetcher"
        aoPayload["OUTPUTS"] = []
        aoPayload["IMAGESCOUNT"] = 0
        wasdi.setPayload(aoPayload)
        wasdi.updateStatus("DONE", 100) 
        return
    # -------------------------------------------------------------------------

    oAllPossiblePeriods = {
        "1hr": timedelta(hours=1),
        "3hr": timedelta(hours=3),
        "6hr": timedelta(hours=6),
        "12hr": timedelta(hours=12),
        "24hr": timedelta(hours=24)
    }

    aoAccumulations = {}
    for sPeriod in [p.strip() for p in sPeriods.split(',')]:
        if sPeriod in oAllPossiblePeriods:
            aoAccumulations[sPeriod] = oAllPossiblePeriods[sPeriod]

    asSelectedGFSFiles = []
    asFinalConvertedOutputs = []
    asTiffImagesAddProcessIds = []
    oProcessToOutputMap = {}

    # 4. Filter and prepare the summation tasks
    for sPeriod, oDelta in aoAccumulations.items():
        oPeriodStart = oTargetTime - oDelta
        asPeriodFiles = filterGFSFilesForPeriod(asGFSHourlyFiles, oPeriodStart, oTargetTime)

        if not asPeriodFiles:
            wasdi.wasdiLog(f"Warning: No GFS images found for accumulation period {sPeriod}")
            continue

        wasdi.wasdiLog(f"Processing {len(asPeriodFiles)} GFS images for {sPeriod} accumulation")

        asSelectedGFSFiles.extend(asPeriodFiles)

        sStartStr = oPeriodStart.strftime("%Y%m%d_%H00")
        sEndStr = oTargetTime.strftime("%Y%m%d_%H00")
        sRawSumFile = f"RAW_SUM_{sBaseName}_{sPeriod}_{sStartStr}-{sEndStr}.tif"
        
        oBBoxSimple = f'{oAlignedBBox["northEast"]["lat"]},{oAlignedBBox["southWest"]["lng"]},{oAlignedBBox["southWest"]["lat"]},{oAlignedBBox["northEast"]["lng"]}'
        aoTiffImagesParams = {"BBOX": oBBoxSimple, "OUTPUT_FILE": sRawSumFile, "INPUT_FILES": asPeriodFiles}

        sProcessId = wasdi.executeProcessor("tiff_images_add", aoTiffImagesParams)
        asTiffImagesAddProcessIds.append(sProcessId)
        oProcessToOutputMap[sProcessId] = (sRawSumFile, sPeriod, sStartStr, sEndStr)

    aoPayload["GFS_FILES"] = sorted(list(set(asSelectedGFSFiles)))    
    wasdi.setPayload(aoPayload)
    wasdi.updateProgressPerc(60)
    # Only wait if there are actually processes to wait for
    if asTiffImagesAddProcessIds:
        wasdi.waitProcesses(asTiffImagesAddProcessIds)
        wasdi.wasdiLog('All tiff_images_add processors finished.')
    wasdi.updateProgressPerc(80)

    # 5. Convert units to match IMERG (mm/s * 3600 -> mm)
    for sProcessId, (sRawSumFile, sPeriod, sStartStr, sEndStr) in oProcessToOutputMap.items():
        sFinalOutputFile = f"{sBaseName}_Cumulative_{sPeriod}_GFS_{sStartStr}-{sEndStr}.tif"
        
        wasdi.wasdiLog(f"Converting units for {sPeriod} accumulation (* 3600) -> {sFinalOutputFile}")
        
        if convertRasterUnits(sRawSumFile, sFinalOutputFile, 3600.0):
            asFinalConvertedOutputs.append(sFinalOutputFile)
            
            wasdi.deleteProduct(sRawSumFile)

    # 6. Cleanup hourly files if requested
    if bDelete:
        wasdi.wasdiLog("Cleaning up intermediate hourly GFS fetcher outputs...") 
        for sFile in asGFSHourlyFiles:
            try:
                wasdi.deleteProduct(sFile)
            except:
                pass

    aoPayload["OUTPUTS"] = asFinalConvertedOutputs
    aoPayload["IMAGESCOUNT"] = len(asFinalConvertedOutputs)
    wasdi.setPayload(aoPayload)
    wasdi.updateStatus("DONE", 100)
    wasdi.wasdiLog(f"END: Processor finished successfully.")


if __name__ == '__main__':
    wasdi.init("./config.json")
    run()