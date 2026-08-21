#
# Module to re-organize FLDPLN library: turn segment-based to tile-based library
#
# imports
from lib2to3.pgen2 import driver
import os
import math
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.io as sio #import loadmat, savemat
import h5py # for reading some .mat file
import glob
from pyproj import CRS
from shapely.geometry import LineString
from osgeo import ogr
import zipfile
import shutil
import urllib.request
import re

# import FLDPLN library header
from fldpln_header import *

############################################################################################################################################
# Functions--convert segments to tiles
############################################################################################################################################
#
# Read matlab files with different versions
#
def ReadMatFile(matFile, varName):
    #scipy.io DOES NOT support matlabe files version 7.3 yet! Some of the libraries are in 7.3 while the others are not.
    try: 
        # load the segment FSP-FPP-DTF table in mat file as < 7.3. 
        vars = sio.loadmat(matFile)
        var = vars[varName]
    except NotImplementedError:
        vars = {}
        f = h5py.File(matFile,'r')
        for k, v in f.items():
            vars[k] = np.array(v)
        # get the variable and tranpose it for dataframe!
        var = vars[varName].transpose()
    except:
        ValueError('Could not read the mat file at all...')
        var = None

    return var
#
# Tile a library. Turn segment-based FSP-FPP relations to tile-based
#
def TileLibrary(segLibFolder,segLibName,tiledLibFolder,tiledLibName,tileSize,fileFormat):
# This function uses the fsp_info.csv file (under tiledLibFolder\tiledLibName) to get FSP IDs
# fileFormat: 'snappy' or 'mat'. 'snappy' format needs to install the 'fastparquet' python package

    # # create the tiledLibFolder folder for all tiled libraries if it doesn't exist
    # os.makedirs(tiledLibFolder,exist_ok=True)

    # # create a folder for the tiled library
    # os.makedirs(os.path.join(tiledLibFolder,tiledLibName),exist_ok=True)

    # Read lib meta data file
    metaDataFile = os.path.join(tiledLibFolder,tiledLibName,metaDataFileName)
    if not os.path.exists(metaDataFile):
        print('Cannot find file: {metaDataFile}')
    else:
        with open(metaDataFile,'r') as jf:
            md = json.load(jf)
        cellSize = md['CellSize']

    # calculate library and segment extents
    libExt, segExts = CalculateLibraryExtent(segLibFolder,segLibName,cellSize)
    minX,maxX,minY,maxY = libExt

    # Calculate tile boundaries
    tb = CalculateTileBoundary(minX,maxX,minY,maxY,tileSize,tileSize)
    print('Number of (possible) tiles:', len(tb))
    print('Tile extents:\n', tb, '\n')

    #
    # Create tiles
    #
    print('Build tiles (tiling FSP-FPP relations) ...')
    hcs = cellSize/2
    # read in FSP IDs
    # need to set float_precision='round_trip' to prevent rounding while reading the text file! float_precision='high' DOESN'T work.
    # For Verdigris 10-m library, FSP ID of 22246, its FspX of -1003.7918248322967 in fsp_info.csv was read into memory as -1003.7918248322968 without using float_precision='round_trip'
    fspIds = pd.read_csv(os.path.join(tiledLibFolder,tiledLibName,fspFileName),float_precision='round_trip',index_col=False)[['FspId','FspX','FspY']]

    # initialize the index DFs and tile ID
    fspIdxDf = pd.DataFrame(columns=fspIdxColumnNames)
    tileIdxDf = pd.DataFrame(columns=tileIdxColumnNames)
    tileId = 1
    for t in tb:
        print('Processing tile: ', tileId)

        # tile boundary, not the cell center boundary!
        tminX, tmaxX,tminY, tmaxY = t
        print('Tile extent (minX, maxX, minY, maxY) :',(tminX, tmaxX,tminY, tmaxY))

        # find the segments that intersect the tile rectangle
        segs = segExts[~((segExts['MinX']>tmaxX) | (segExts['MaxX']<tminX))] # rectangles are NOT on left or right of each other
        segs = segs[~((segs['MinY']>tmaxY) | (segs['MaxY']<tminY))] # rectangles are NOT on top or bottom of each other
        segs = segs['FileName'].to_list()
        print('Number of segments interseting with the tile: ', len(segs))

        # find the FPP-FSP relations in the tile from all the intersecting segments
        # Initialize the df
        tdf = pd.DataFrame()
        for mf in segs: #segMatFileFullNames:
            # read mat file
            matVar = ReadMatFile(mf,matRelVarName)
            # converting array to a dataframe
            sdf = pd.DataFrame(matVar, columns=matRelColumnNames)

            # rename columns to better names
            betterColumnNames = ["FspX", "FspY", "FppX", "FppY", "Dtf", "FilledDepth"]
            d=dict(zip(matRelColumnNames,betterColumnNames))
            sdf.rename(columns=d,inplace=True) # inplace changing column name
        
            # Code used to find out which segment file (e.g., SLIE_segment40.mat) in library 'midkan" has problem 
            # where column “DTF + fill depth” is LESS than the "DTF" column
            # t  =  sdf['FilledDepth'] - sdf['Dtf']
            # t  =  t < -0.1
            # if t.any():
            #     print('Segment: ', mf)
            #     print(f'Found negative depression in {mf}!')

            # select the FPPs within the tile
            sdf = sdf[(sdf['FppX']>=tminX) & (sdf['FppX']<=tmaxX) & (sdf['FppY']>=tminY) & (sdf['FppY']<=tmaxY)].copy() # tell pandas we want a copy to avoid "SettingWithCopyWarning" in line 86 when there is only one segment mat file read
            # print('Segment:', segId, 'Number of FSP-FPP relations:', len(sdf))
            if tdf.empty:
                tdf = sdf
            else:
                tdf = tdf.append(sdf)
        print('Total number of FSP-FPP relations in the tile:', len(tdf))
               
        # save the FSP-FPP relations and update the index dfs
        if (not (tdf.empty)) and (len(tdf) != 0): # there are some FPPs in the tile
            #calculate FilledDepth
            tdf['FilledDepth'] = tdf['FilledDepth']-tdf['Dtf']

            # Calculate FSP center extent for the tile
            fspMinX = tdf['FspX'].min()
            fspMaxX = tdf['FspX'].max()
            fspMinY = tdf['FspY'].min()
            fspMaxY = tdf['FspY'].max()
            
            # Calculate FPP center extent within the tile
            fppMinX = tdf['FppX'].min()
            fppMaxX = tdf['FppX'].max()
            fppMinY = tdf['FppY'].min()
            fppMaxY = tdf['FppY'].max()
            
            # turn FPP coordinates into row and column within the tile
            # Note that the row and column start at (fppMinX, fppMaxY)!
            tdf['FppX'] = ((tdf.FppX-fppMinX)/cellSize).round()
            tdf['FppY'] = ((fppMaxY-tdf.FppY)/cellSize).round()
            # tdf['FppIdx'] = tdf.FppX * tRows + tdf.FppY

            # rename ['FppX','FppY'] to ['Col','Row']
            tdf.rename(columns={'FppX':'FppCol','FppY':'FppRow'},inplace=True)

            # merge relations with FSP IDs
            tdf = tdf.merge(fspIds,how='left',on=['FspX','FspY'])

            # # check FSP IDs
            # if tdf.isnull().values.any():
            #     print('There are NAN in the dataframe!')

            # remove FspX & FspY
            tdf.drop(['FspX','FspY'],axis=1,inplace=True)

            # reorder columns
            tdf = tdf[relColumnNames]

            # set datatypes for the columns
            tdf=tdf.astype(dtype={"FspId":np.int32, "FppCol":np.int32, "FppRow":np.int32, "Dtf":np.float32, "FilledDepth":np.float32},copy=False)
            # convert float64 to float32 before saving and create index on FspX and FspY for speed up merge during mapping
            # tdf.astype(np.float32,copy=False).set_index(keys=['FspX','FspY'],inplace=True)

            # save the relations in the tile in a file
            print('Saving FSP-FPP relations in a file...')
            if fileFormat == 'snappy':
                # filePathName = os.path.join(segLibFolder, tiledLibName, tileFileMainName+'_'+str(tileId)+'.gzip') # for gzip format
                filePathName = os.path.join(tiledLibFolder, tiledLibName, tileFileMainName+'_'+str(tileId)+'.snz') # for snappy format
                
                # save to parquet file. Can only save one DataFrame!
                # tdf.to_parquet(filePathName,engine='fastparquet',compression='gzip',index=False) # gzip
                tdf.to_parquet(filePathName,engine='fastparquet',compression='snappy',index=False) # snappy fast processing
                # tdf.to_parquet(filePathName,engine='fastparquet',compression='snappy',index=True) # snappy fast processing
            elif fileFormat == 'mat':
                # seperate columns by datatypes (int32 and float32)
                fspFppsArray = tdf[relColumnNames[0:3]].to_numpy(dtype=np.int32)
                dtfFilledDepthArray = tdf[relColumnNames[-2::]].to_numpy(dtype=np.float32)
                # Save to compressed .mat file
                dfDic = {'FspFpps': fspFppsArray,'DtfFilledDepth': dtfFilledDepthArray}
                # Tile cannot be too large which may cause failture in writing into .mat file. See https://github.com/scipy/scipy/issues/12465
                filePathName = os.path.join(tiledLibFolder, tiledLibName, tileFileMainName+'_'+str(tileId)+'.mat')
                sio.savemat(filePathName, dfDic, do_compression=True) 
            else:
                print('Unsupported file format!')
                return

            # calculate the min and max DTF for each FSPs in the tile
            fspDf = tdf.groupby(['FspId'], as_index=False).agg(MinDtf = ('Dtf', min),MaxDtf = ('Dtf', max))
            print('Number of unique FSPs in the tile:', len(fspDf))
            # print(fspDf)
            
            # add tile ID to fsp-tile index
            fspDf['TileId'] = tileId
            # reorder columns
            fspDf = fspDf[fspIdxColumnNames]
            # append to the index table
            fspIdxDf = fspIdxDf.append(fspDf,ignore_index=True)
            # print('Fsp-Tile index for the tile:')
            # print(fspIdxDf)

            # Calculate FSP & FPP external extents for saving in the tile-index file
            fspMinX,fspMaxX,fspMinY,fspMaxY = fspMinX-hcs,fspMaxX+hcs,fspMinY-hcs,fspMaxY+hcs
            fppMinX,fppMaxX,fppMinY,fppMaxY = fppMinX-hcs,fppMaxX+hcs,fppMinY-hcs,fppMaxY+hcs
            print('Tile FSP extent (fspMinX,fspMaxX,fspMinY,fspMaxY): ', (fspMinX,fspMaxX,fspMinY,fspMaxY))
            print('Tile FPP extent (fppMinX,fppMaxX,fppMinY,fppMaxY): ', (fppMinX,fppMaxX,fppMinY,fppMaxY))

            # Calculate the min & max DTF within the tile
            minTileDtf = fspDf['MinDtf'].min()
            maxTileDtf = fspDf['MaxDtf'].max()

            # calculate number of relations and number of FPPs in the tile
            numOfRels = len(tdf)
            numOfFpps = len(tdf[['FppCol','FppRow']].drop_duplicates()) # groupby for fast?
            
            # add tile ID and additional tile-related info to the tile index file
            tileIdx = pd.DataFrame([[tileId,tminX,tmaxX,tminY,tmaxY,fppMinX,fppMaxX,fppMinY,fppMaxY,fspMinX,fspMaxX,fspMinY,fspMaxY,minTileDtf,maxTileDtf,numOfRels,numOfFpps]],columns=tileIdxColumnNames)
            # append to the index table
            tileIdxDf = tileIdxDf.append(tileIdx,ignore_index=True)
            # print('Tile index for the tile:')
            # print(tileIdxDf)

            # move to the next tile
            tileId +=1

    # Save the index as a file
    # print('Number of items in the fsp-tile index:', len(fspIdxDf))
    # print('fsp-tile Index table:\n',fspIdxDf)
    # save index to a csv file
    print('Save fsp-tile index as a CSV file ...')
    fspIdxDf.to_csv(os.path.join(tiledLibFolder, tiledLibName, tileFileMainName+'_fsp_index.csv'),index=False)

    # print('Number of items in the tile-extent index:', len(tileIdxDf))
    # print('Tile index table:\n',tileIdxDf)
    # save index to a csv file
    print('Save tile index as a CSV file ...')
    tileIdxDf.to_csv(os.path.join(tiledLibFolder, tiledLibName, tileFileMainName+'_tile_index.csv'),index=False)

    return

#
# Calculate each tile's boundary as (minX, maxX,minY, maxY)
#
# minX, maxX, minY, maxY are the external border (not the cell center) coordinates of the area needs to be tiled.
# tileSizeX, tileSizeY are also the external border size of a tile, not the cell center size!
def CalculateTileBoundary(minX, maxX, minY, maxY, tileSizeX, tileSizeY, padding=True):
#padding--all the tiles have the same size, not reduced to the border of the tiled area
    def TileInOneDimension(min, max, tileSize, padding=True):
        # generate tile marker coordinates
        bs = np.arange(min, max, tileSize)
        # handle the last coordinate as np.arange() is inclusive
        if padding: #all the tiles have the same length whether it's outside the border of the tiled area or not
            bs = np.append(bs, bs[-1]+tileSize)
        else: # the last tile stops at the border of the area
            bs = np.append(bs, max)

        # generate tile border coordinate pairs
        tb = [(bs[t], bs[t+1]) for t in range(bs.size-1)]
        return tb
        
    xb = TileInOneDimension(minX,maxX,tileSizeX,padding)
    yb = TileInOneDimension(minY,maxY,tileSizeY,padding)
    tb = [(x, y) for x in xb for y in yb] # in ((minX, maxX),(minY, maxY))
    # convert to (minX, maxX,minY, maxY)
    tb = [(minX, maxX,minY, maxY) for ((minX,maxX),(minY,maxY)) in tb]
    return tb

#
# calculate library external border extent
#
def CalculateLibraryExtent(segLibFolder,segLibName,cellSize):
# return external border extent (minX, maxX,minY, maxY)
# and segment extents (FPP cell center) df ['MinX','MaxX','MinY','MaxY','FileName']

    # Get all the segment mat files in the library/watershed
    segMatFileFullNames = glob.glob(os.path.join(segLibFolder, segLibName, segMatFileMainName+'*.mat'))
    hcs = cellSize/2

    # initialize extent
    minX,maxX,minY,maxY = (math.inf,-math.inf,math.inf,-math.inf)
    relNum = 0 # number of FSP-FPP relation
    segExts = pd.DataFrame(columns=['MinX','MaxX','MinY','MaxY','FileName']) # empty df storing each segment's FPP extent and correponding mat file name
    # update the extent by all the segments
    print('Calculate library extent ...')
    for mf in segMatFileFullNames:
        # read mat file
        matVar = ReadMatFile(mf,matRelVarName)
        # converting array to a dataframe
        df = pd.DataFrame(matVar, columns=matRelColumnNames)

        relNum += len(df)
        # print('Segment:', segId, 'Number of FSP-FPP relations:', len(df))

        # Calculate segment external border extent
        sminX = df['floodplain_pixel_x'].min()-hcs
        sminY = df['floodplain_pixel_y'].min()-hcs
        smaxX = df['floodplain_pixel_x'].max()+hcs
        smaxY = df['floodplain_pixel_y'].max()+hcs
        segExt= pd.DataFrame([[sminX,smaxX,sminY,smaxY,mf]],columns=['MinX','MaxX','MinY','MaxY','FileName'])
        # add the segment extent to the df
        segExts = segExts.append(segExt)

        #update library extent
        if sminX < minX: minX = sminX
        if sminY < minY: minY = sminY
        if smaxX > maxX: maxX = smaxX
        if smaxY > maxY: maxY = smaxY
 
    print('Library external border extent (minX, maxX, minY, maxY) :',(minX, maxX,minY, maxY))
    # print('Library segment FPP extents:\n', segExts)
    print('Total number of FSP-FPP relations:', relNum)
    
    return (minX, maxX,minY, maxY), segExts

############################################################################################################################################
# Functions--clean up and add additional files to tiled library
############################################################################################################################################

def CreateFspSegmentInfoSpatialReferenceFiles(libFolder,libName,cellSize,prjFileName,outLibFolder,outLibName):
# Library's spatial reference is saved in shapefile's .prj file
# Cleanup segments (some segments don't exist in FSPs) and save library FSP and segment information as two csv files 
# (fsp_info.csv & segment_info.csv). It also reads in the SpatialReference.prj and save it in CellSizeSpatialReference.json
# It also calculates FSP and segment downstream distance (i.e., distance to library outlet(s)) which involves:
# 1. Calculate FSP's within-segment downstream distance
# 2. Calculate segment length which is more accurate than "CellCount" * cell size
# 3. Calculate segment's dowstreeam distance (to watershed outlet) for speeding up 
# 4. Calculate FSP's downstream distance
# Note that FSPs and segments are based on raster cell centers. Segment and its downstream segment has a gap (1 cell or sqrt(2) cell).
# This function add the gap when calculating downstream distance to the outlet!

    # Original FSP info and segment network Excel files
    fspInfoFileName = fspInfoFileNameHeader+libName+fspInfoFileNameTail
    segInfoFileName = segInfoFileNameHeader+libName+segInfoFileNameTail
    
    #
    # read in fsp (flood source pixel) and segment network Excel files
    #
    fspExcelFile = os.path.join(libFolder, libName, fspInfoFileName) #'SLIE_KS_AOI_'+libName+'_fsp_info.xlsx')
    segExcelFile = os.path.join(libFolder, libName, segInfoFileName) #'segment_network_info_'+libName+'.xlsx')
    # fspDf = pd.read_excel(fspExcelFile, sheet_name=fspInfoSheetName, engine='openpyxl') # need to specify engine when using python env "geo3"!
    fspDf = pd.read_excel(fspExcelFile, sheet_name=fspInfoSheetName)
    # print(fspDf)
    # segDf = pd.read_excel(segExcelFile, sheet_name=segInfoSheetName, engine='openpyxl') # need to specify engine when using python env "geo3"!
    segDf = pd.read_excel(segExcelFile, sheet_name=segInfoSheetName)
    # print(segDf)

    #
    # Clean up the segment table.
    # 1. Remove the segment if it's not in the FSP table
    # 2. If the segment is the downstream segment of another segment, set it as 0. 
    # Those missing segments are usually because of they are close to or in waterbodies. 
    # By removing those segment, a library may have several seperate watersheds/outpets!
    #
    # get the segment IDs
    segIds = segDf['segID'].to_list()
    for sid in segIds:
        fsps = fspDf[fspDf['SegID']==sid]
        if len(fsps)==0:
            # segment not found in the FSP table. delete the row
            segDf = segDf.loc[segDf['segID']!=sid]
            # set downstream segment ID to 0
            segDf.loc[segDf['segID_ds']==sid,'segID_ds'] = 0

    #
    # Calculate FSP within-segment distance, segment length, and segment dowstream distance, and FSP downstream distance
    #
    # add field for FSP within-segment distance
    fspDf['DsDist'] = 0.0
    # add field for segment length
    segDf['Length'] = 0.0

    # Calculate FSP within-segment DOWNSTREAM distance and segment length
    for segIdx, row in segDf.iterrows():
        segID = row['segID']
        # print(segID)

        # select FSP on the segment
        fsps = fspDf[fspDf['SegID']==segID][['X','Y']]
        
        # calculate fsp downstream within segment length
        segDist = 0.0
        if len(fsps)==0:
            # this should not happen as we have already clean up the segment table!
            print(f"Segment {segID} is missing in {fspFileName}!")
        else:  
            first=True
            for idx, row in fsps[::-1].iterrows():
                # Note the idx in fsps is the index in fspDf!!!
                # calculate distance
                if first:
                    fspx1, fspy1 = row['X'], row['Y']
                    dist=0.0
                    first=False
                else:
                    fspx2, fspy2 = row['X'], row['Y']
                    dist=math.sqrt((fspx1-fspx2)**2+(fspy1-fspy2)**2)
                    fspx1, fspy1 = fspx2, fspy2
                segDist += dist
                fspDf.at[idx,'DsDist']=segDist

        # update segment distance in segDf
        segDf.at[segIdx,'Length'] = segDist

    # show the DFs
    # print(fspDf[:1136])
    # print(segDf)

    #
    # Calculate segment downstream length
    #
    # only for the segments that exist in the FSP table
    # But this not necessary as segments don't exist in FSP table already removed
    # Also this line will remove the segment which has just one FSP!
    # segDf = segDf[segDf['Length']>0]

    # add field for segment downstream distance for speeding up calculating FSP downstream distance
    segDf['DsDist'] = 0.0
    for segIdx, row in segDf.iterrows():
        segID = row['segID']
        dsSegID = row['segID_ds']

        dsDist = 0.0
        while dsSegID != 0:
            # get downstream segment length and ID
            tempDf = segDf[segDf['segID']==dsSegID][['Length','segID_ds']]
            length, segID_ds = tempDf.iat[0,0], tempDf.iat[0,1]
            dsDist += length

            # There is a GAP between two segments as they are consisted of FSP cell centers
            # Calculate the GAP and add it to segment downstream dist
            # last fsp in upstream segment
            lastFspXy = fspDf[fspDf['SegID']==segID][['X','Y']].tail(1)    
            fspx1, fspy1 = lastFspXy.iat[0,0],lastFspXy.iat[0,1]
            # first FSP in downstream segment
            firstFspXy = fspDf[fspDf['SegID']==dsSegID][['X','Y']].head(1)
            fspx2, fspy2 = firstFspXy.iat[0,0], firstFspXy.iat[0,1]
            dist=math.sqrt((fspx1-fspx2)**2+(fspy1-fspy2)**2)
            dsDist += dist

            # move to ownstream segment
            segID = dsSegID
            dsSegID = segID_ds

        segDf.at[segIdx,'DsDist'] = dsDist
    # print(segDf)

    # Calculate FSP downstream distance
    for idx, row in fspDf.iterrows():
        segID = row['SegID']
        inSegDist = row['DsDist']

        # get segment downstream distance
        tempDf = segDf[segDf['segID']==segID][['DsDist']]
        segDsDist = tempDf.iat[0,0]

        # reset FSP downstream distance
        fspDf.at[idx,'DsDist'] = inSegDist + segDsDist
    # print(fspDf)

    #
    # Standardize columns before saving to files
    #
    # for FSPs
    # original column names: ['X', 'Y', 'StrID', 'SegID', 'FIL', 'DsDist']
    # change to: fspColumnNames = ['FspX','FspY','SegId','FilledElev','DsDist','StrOrd']
    # drop off 'StrID'
    fspDf.drop(['StrID'],axis=1,inplace=True)
    # rename the columns
    fspDf.rename(columns={'X':'FspX','Y':'FspY','SegID':'SegId','FIL':'FilledElev'},inplace=True)
    # add unique ID to FSPs
    fspDf['FspId'] = fspDf.index
    # reorder the columns
    fspDf = fspDf[fspColumnNames[:-1]] # 'StrOrd' column will be added later on

    # for segments
    # original column names: ['segID', 'seg_len', 'segID_ds', 'Length', 'DsDist']
    # change to: segColumnNames = ['SegId','CellCount','DsSegId','Length','DsDist','StrOrd']
    # rename the columns
    # rename segment ID column to make them the same in two files for easy join
    segDf.rename(columns={'segID':'SegId','seg_len':'CellCount','segID_ds':'DsSegId'},inplace=True)

    # save the DFs in CSVs
    # create the outLibFolder folder for all tiled libraries if it doesn't exist
    os.makedirs(outLibFolder,exist_ok=True)

    # create a folder under outLibFolder for the tiled library
    os.makedirs(os.path.join(outLibFolder,outLibName),exist_ok=True)

    fspCsvFile = os.path.join(outLibFolder, outLibName, fspFileName)
    segCsvFile = os.path.join(outLibFolder, outLibName, segFileName)
    fspDf.to_csv(fspCsvFile,index=False)
    segDf.to_csv(segCsvFile,index=False)

    #
    # save cell size and spatial reference in a metadata file 
    #
    # read in spatial reference for the library
    srFile = os.path.join(libFolder,prjFileName)
    with open(srFile, 'r') as srf:
        srText = srf.read()
    cellSizeSr = {'CellSize': cellSize, 'SpatialReference': srText}
    # save metedata
    with open(os.path.join(outLibFolder, outLibName, metaDataFileName), 'w') as jf:
        json.dump(cellSizeSr,jf)

    return fspDf, segDf, cellSizeSr

#
# Generate ONE segment shapefile for a list of libaries for manually assigning stream order which is used in FSP DOF interpolation
#
def GenerateSegmentShapefiles(inLibFolder, inLibNames, outFolder, outShpFileName):
# Generate segment shapefiles for a list of libraries.

    if len(inLibNames)==0:
        print('No library!')
        return None

    # create an empty geodataframe for all the libraries
    allLibSegs = gpd.GeoDataFrame(columns = segShpColumnNames)
    allLibSegs = allLibSegs.astype(dtype={"LibName": 'string', "SegId": 'int32', "StrOrd": 'int32'})

    for libName in inLibNames:
        # read fsp and segment info csv files
        fspDf = pd.read_csv(os.path.join(inLibFolder,libName,fspFileName),index_col=False)
        segDf = pd.read_csv(os.path.join(inLibFolder,libName,segFileName),index_col=False)
        
        # create an empty geodataframe for the library
        libSegs = gpd.GeoDataFrame(columns = segShpColumnNames)
        libSegs = libSegs.astype(dtype={"LibName": 'string', "SegId": 'int32', "StrOrd": 'int32'})

        # 
        # Populate the gpd with segments. Create a line for each segment
        #
        idx = 0
        for segIdx, row in segDf.iterrows():
            segID = row['SegId']
            dsSegID = row['DsSegId']

            # select FSP on the segment
            fsps = fspDf[fspDf['SegId']==segID][['FspX','FspY']]
                
            # There is a GAP between two segments as they are consisted of FSP cell centers
            # Upstream sgements are EXTENDED to the first FSP of the downstream segment!
            if dsSegID != 0:
                # first FSP in downstream segment
                firstFspXy = fspDf[fspDf['SegId']==dsSegID][['FspX','FspY']].head(1)
                # append to the fsps
                fsps = fsps.append(firstFspXy)
            # print(fsps)

            # turn the FSPs into a LineString
            # create Points, a GeonetryArray, from fsps
            points = gpd.points_from_xy(fsps['FspX'],fsps['FspY'])
            segLineStr = LineString(points)
            # print(segLineStr)

            # Insert the line into geodataframe
            libSegs.at[idx, 'geometry'] = segLineStr
            # Add additional columns
            libSegs.at[idx, 'LibName'] = libName
            libSegs.at[idx, 'SegId'] = segID
            idx += 1
                
        # append the segments to all lib segments
        allLibSegs = allLibSegs.append(libSegs,ignore_index=True)

    # Read lib meta data file
    metaDataFile = os.path.join(inLibFolder, libName, metaDataFileName)
    with open(metaDataFile,'r') as jf:
        md = json.load(jf)
    srText = md['SpatialReference']
    # set geodataframe CRS
    libSr = CRS.from_wkt(srText)
    allLibSegs.crs = libSr

    # Write the data into that Shapefile
    schema = gpd.io.file.infer_schema(allLibSegs)
    schema['properties']['LibName'] = 'str:32'
    schema['properties']['SegId'] = 'int32:10'
    schema['properties']['StrOrd'] = 'int32:10'
    shpFullName = os.path.join(outFolder, outShpFileName)
    allLibSegs.to_file(shpFullName, driver= "ESRI Shapefile",schema=schema) # existing shapefile will be replaced automatically!!!

    return allLibSegs

#
# Get stream order for FSPs and segments from sgement shapefile and generate stream-order network information for FSP DOF interpolation
#
def GetStreamOrdersForFspsSegments(libFolder,libName,strOrdShpFolder,strOrdShpName):
# Updates fsp_info.csv and segment_info.csv files with stream order
# Craetes the stream_order_info.csv which stores the network info at the level of stream orders
        
    # get stream order from feature class in geodatabase
    strOrdShpFullName = os.path.join(strOrdShpFolder,strOrdShpName) 
    shpDf = gpd.read_file(strOrdShpFullName)
    
    # select the segments in the library
    segOrdDf = shpDf[shpDf['LibName']==libName]
    # select columns
    streamOrderColumns = ['SegId','StrOrd']
    segOrdDf = segOrdDf[streamOrderColumns]
   
    # read fsp and segment info csv files
    fspDf = pd.read_csv(os.path.join(libFolder,libName,fspFileName),index_col=False)
    segDf = pd.read_csv(os.path.join(libFolder,libName,segFileName),index_col=False)

    # remove existing "StrOrd" column
    if ('StrOrd' in fspDf.columns):
        fspDf.drop(['StrOrd'], axis=1,inplace=True)
    if ('StrOrd' in segDf.columns):
        segDf.drop(['StrOrd'], axis=1,inplace=True)

    # Get segment stream order by merging DFs based on segment ID
    segDf = pd.merge(segDf, segOrdDf, how='left', on='SegId')
    # print(segDf)

    # Get FSP's stream order by merging
    fspDf = pd.merge(fspDf, segOrdDf, how='left', on='SegId')
    # print(segDf)

    # save the DFs in CSVs
    fspCsvFile = os.path.join(libFolder,libName,fspFileName)
    segCsvFile = os.path.join(libFolder,libName,segFileName)
    fspDf.to_csv(fspCsvFile,index=False)
    segDf.to_csv(segCsvFile,index=False)

    #
    # Create another table storing stream order network information with the columns:
    # [‘StrOrd’, ‘DsStrOrd’, ‘JunctionFspX’, ‘JunctionFspY’] for use in interpolting FSP DOF
    #
    strOrdDf = pd.DataFrame(columns=strOrdColumnNames)
    strOrds = segDf['StrOrd'].drop_duplicates().sort_values().to_list()
    for so in strOrds:
        # find the most downstream segment in the stream order
        mostDsSeg = segDf[segDf['StrOrd']==so].sort_values('DsDist')[['DsSegId']].head(1)
        # print(mostDsSeg)
        # get its downstream segment ID
        dsSegId = mostDsSeg.iat[0,0]
        if dsSegId != 0:
            # get downstream stream order
            dsOrd = segDf[segDf['SegId']==dsSegId]['StrOrd'].iat[0]
            # get the first FSP in the downstream segment
            firstFspXy = fspDf[fspDf['SegId']==dsSegId][['FspX','FspY']].head(1)
            fspx, fspy = firstFspXy.iat[0,0], firstFspXy.iat[0,1]
        else:
            # no downstream segment
            dsOrd,fspx,fspy = 0,0,0

        # add the connectivity information to the table
        temp = pd.DataFrame([[so,dsOrd,fspx,fspy]],columns=strOrdColumnNames)    
        # append to the index table
        strOrdDf = strOrdDf.append(temp,ignore_index=True)
    
    # save the table as a CSV file
    strOrdCsvFile = os.path.join(libFolder, libName, strOrdFileName)
    strOrdDf.to_csv(strOrdCsvFile,index=False)

    return fspDf, segDf, strOrdDf

#
# A function to download and unzip segment-based libraries
#
def DownloadSegmentLibrary(segUrl,libName,localLibFolder ):
# Not implemented yet! The zip file name may be different in the libraries. 
    # 
    # get the zipped segment library file name, which can vary from library to libaray as the max DOF may be different
    # for example: SLIE_KS_AOI_spring_DTF_49ft_(15m).zip, SLIE_KS_AOI_ninnescah_DTF_33ft_(10m).zip	
    #
    segUrl = segUrl+'/'+libName
    # remove space in the URL
    url = segUrl.replace(" ","%20")
    # create a request
    req = urllib.request.Request(url)
    # open and read the HTML page
    htmlPage = urllib.request.urlopen(req).read()
    # turn the HTML page from byte into string
    htmlPageStr = str(htmlPage)

    # get segment library zip file from the HTML page
    p1 = 'href=\"SLIE_KS_AOI_'+libName+'_DTF_\d{,3}ft_\(\d{,3}m\)\.zip\"'
    hrefStr = re.findall(p1,str(htmlPageStr))[0]
    p2 = 'SLIE_KS_AOI_'+libName+'_DTF_\d{,3}ft_\(\d{,3}m\)\.zip'
    zipFile = re.findall(p2,hrefStr)[0]
    
    # create local folder if not existing
    os.makedirs(localLibFolder,exist_ok=True)

    # If you need to redownload for whatever reason
    if os.path.exists(os.path.join(localLibFolder, libName)):
        shutil.rmtree(os.path.join(localLibFolder, libName))
    os.mkdir(os.path.join(localLibFolder, libName))

    # Download base library
    print(f'Downloading library {libName} ...')
    urllib.request.urlretrieve(segUrl+'/'+zipFile,os.path.join(localLibFolder, libName, zipFile))
    
    # Download FSP info Excel file
    print("Downloading FSP Excel file ...")
    urllib.request.urlretrieve(segUrl+'/SLIE_KS_AOI_'+libName+'_fsp_info.xlsx', 
        os.path.join(localLibFolder, libName, 'SLIE_KS_AOI_'+libName+'_fsp_info.xlsx'))

    # Download segment info Excel file
    print("Downloading segment Excel file ...") #	segment_network_info_spring.xlsx
    urllib.request.urlretrieve(segUrl+'/segment_network_info_'+libName+'.xlsx', 
        os.path.join(localLibFolder, libName, 'segment_network_info_'+libName+'.xlsx'))

    # unzip library
    print("Unzipping ...")
    # both shutil and zipFile do NOT support "Deflated64" or "compression type 9"
    with zipfile.ZipFile(os.path.join(localLibFolder, libName, zipFile),'r') as zip_ref:
        zip_ref.extractall(os.path.join(localLibFolder, libName)) 
    # shutil.unpack_archive(os.path.join(localLibFolder, libName, zipFile),os.path.join(localLibFolder, libName),'zip')

    # clean up folder
    os.remove(os.path.join(localLibFolder, libName, zipFile))
    print("Done for "+libName)
    return
