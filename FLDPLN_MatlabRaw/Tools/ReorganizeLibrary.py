# imports
import time

# fldpln module
from fldpln_reorg import *

########################################################################################################
# Main script
########################################################################################################

# # Test library setttings
# inLibFolder = r'E:\FLDPLN\KS_original_library'
# cellSize = 5
# prjFileName = 'SpatialReference.prj' # library's spatial reference saved in shapefile's .prj file
# tileSize = 10000 
# # outLibFolder = r'E:\FLDPLN\test\KS_tiled10km_mat_library'
# # tileFileFormat = 'mat' # 'snappy' or 'mat'
# # inLibNames = ['spring'] #['lowark','neosho']

# # Production settings
# # outLibFolder = r'E:\FLDPLN\KS_tiled10km_mat_library'
# outLibFolder = r'E:\FLDPLN\KS_tiled10km_snz_library'
# tileFileFormat = 'snappy' # 'snappy' or 'mat'
# allLibNames = [ 'bigblue','chik_saltfork_medlodg','cotneo','ksrv','litark',
#                 'lowark','lowrep','mdc','midkan','midsol',
#                 'morv','nemaha','neosho','ninnescah','osagemarmaton',
#                 'smoksalsol','spring','upark','upkan','uprep',
#                 'upsal','upsmoky','upsol','verdigriscaney','walnut']
# inLibNames =  allLibNames
# #
# # Clean up FSP and segment and tile libraries
# #
# for inLibName in inLibNames:
#     outLibName = inLibName 
    
#     # Create FSP and segment info files and library meta file (i.e., cell size and spatial reference)
#     print(f'Create FSP and segment info and library meta files for library: {outLibName} ...')
#     CreateFspSegmentInfoSpatialReferenceFiles(inLibFolder,inLibName,cellSize,prjFileName,outLibFolder,outLibName)

#     # tile library
#     print(f'Tile library: {outLibName} ...')
#     TileLibrary(inLibFolder,inLibName,cellSize,outLibFolder,outLibName,tileSize,tileFileFormat)

# #
# # Generate segment shapefile for libraries
# #
# inLibFolder = r'E:\FLDPLN\KS_tiled10km_mat_library'
# outFolder = r'E:\FLDPLN\stream_order'
# # allLibNames = [ 'bigblue','chik_saltfork_medlodg','cotneo','ksrv','litark',
# #                 'lowark','lowrep','mdc','midkan','midsol',
# #                 'morv','nemaha','neosho','ninnescah','osagemarmaton',
# #                 'smoksalsol','spring','upark','upkan','uprep',
# #                 'upsal','upsmoky','upsol','verdigriscaney','walnut']
# inLibNames = ['bigblue','morv']
# outShpName = "bigblue_morv.shp" # "segment_stream_order.shp"

# # generate segment shapefile 
# print(f'Generate segment shapefile for libraries ...')
# GenerateSegmentShapefiles(inLibFolder,inLibNames,outFolder,outShpName)

#
# Manually assign stream order to segments, i.e., populate the "StrOrd" field in the shapefile
# This is a MANUAL work !!!
#

#
# Get stream order for FSPs and segments; create stream order network
#
# libFolder = r'E:\FLDPLN\test\KS_tiled10km_mat_library'
# libFolder = r'E:\FLDPLN\KS_tiled10km_mat_library'

###############################################################
# Edit-Junho
###############################################################
#libFolder = r'E:\FLDPLN\KS_tiled10km_snz_library'
libFolder = r'C:\Users\jsong33\Desktop\Research\FIM\FLDPLN_FIM\Amite_h15_dh1_Resample\ARB_tiled10km_snz_library'

#streamOrderFolder = r'E:\FLDPLN\stream_order'
streamOrderFolder = r'C:\Users\jsong33\Desktop\Research\FIM\FLDPLN_FIM\Amite_h15_dh1_Resample\Stream_order'

#streamOrderShpFile = 'segment_stream_order.shp'
streamOrderShpFile = 'library_segments.shp'

#allLibNames = [ 'bigblue','chik_saltfork_medlodg','cotneo','ksrv','litark',
#                'lowark','lowrep','mdc','midkan','midsol',
#                'morv','nemaha','neosho','ninnescah','osagemarmaton',
#                'smoksalsol','spring','upark','upkan','uprep',
#                'upsal','upsmoky','upsol','verdigriscaney','walnut']
allLibNames = ['Amite']

libNames = allLibNames
for libName in libNames:
    print(f'Get stream order and generate stream order network info for library: {libName} ...')
    GetStreamOrdersForFspsSegments(libFolder,libName,streamOrderFolder,streamOrderShpFile)
