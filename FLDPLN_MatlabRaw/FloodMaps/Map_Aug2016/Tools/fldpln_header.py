#
# module provides basic information on FLDPLN library
#
###############################################################################################################
# Segment-based library file information. Only need when turn segment-based library to tile-based library
###############################################################################################################
# segment mat file main name
segMatFileMainName = 'SLIE_segment'
# columns in the segment .mat file, can also be read from the mat file
matRelVarName='fldpln' # variable name storing the FSP-FPP relations in the mat file
matRelColumnNames = ["FSP_x", "FSP_y", "floodplain_pixel_x", "floodplain_pixel_y", "DTF", "DTF_fill_depth"]

# FSP info Excel file: "SLIE_KS_AOI_{libName}_fsp_info.xlsx"
fspInfoFileNameHeader = "SLIE_KS_AOI_"
fspInfoFileNameTail = "_fsp_info.xlsx"
fspInfoSheetName = 'FSP'
fspInfoSheetColumnNames= ['X', 'Y', 'StrID', 'SegID', 'FIL', 'DsDist']

# Segment network info Excel file: "segment_network_info_{libName}.xlsx"
segInfoFileNameHeader = "segment_network_info_"
segInfoFileNameTail = ".xlsx"
segInfoSheetName = 'Sheet1'
segInfoSheetColumnNames = ['segID', 'seg_len', 'segID_ds', 'Length', 'DsDist']

############################################################################################################
# Tile-based library file information. Need to map a library
############################################################################################################
# meta data: cell size and spatial reference
metaDataFileName = 'CellSizeSpatialReference.json'

# tile file main name and FSP-FPP relation columnNames
tileFileMainName = 'FLDPLN_tiled'
relColumnNames = ["FspId", "FppCol", "FppRow", "Dtf", "FilledDepth"]

# FSP-tile index file and its column names
fspTileIndexFileName = f"{tileFileMainName}_fsp_index.csv"
fspIdxColumnNames=['TileId','FspId', 'MinDtf', 'MaxDtf'] # DTF--"FSP's depth" to flood a FPP

# Tile index file and its column names
tileIndexFileName = f"{tileFileMainName}_tile_index.csv"
tileIdxColumnNames=['TileId','TileMinX','TileMaxX','TileMinY','TileMaxY','FppMinX','FppMaxX','FppMinY','FppMaxY','FspMinX','FspMaxX','FspMinY','FspMaxY','MinDtf','MaxDtf','NumOfRels','NumOfFpps']

# FSP info file 
fspFileName = "fsp_info.csv"
fspColumnNames = ['FspId','FspX','FspY','SegId','FilledElev','DsDist','StrOrd']

# Segment network info file
segFileName = "segment_info.csv"
segColumnNames = ['SegId','CellCount','DsSegId','Length','DsDist','StrOrd']

# Segment shapefile columns for manually assign segment stream order
segShpColumnNames = ['geometry','LibName','SegId','StrOrd'] # stream order has to be defined manually!

# Stream-order network information file for DOF interpolation
strOrdFileName = 'stream_order_info.csv'
strOrdColumnNames = ['StrOrd', 'DsStrOrd', 'JunctionFspX', 'JunctionFspY']
