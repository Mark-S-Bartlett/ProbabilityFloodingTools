# Python 3.6

#---------------------------------------------------------------------------#
#-- Import Modules/Libraries -----------------------------------------------#
#---------------------------------------------------------------------------#
import os, sys, gdal
import osr
import pandas as pd
import numpy as np
import geopandas as gpd
import pathlib as pl

from osgeo import gdal
import rasterio
from io import BytesIO
import json

import boto3
s3 = boto3.resource('s3')

class Project:

    def __init__(self, projName:str, modelName:str, bucket:str='pfra', local_path:str=''):
        self._project_name = projName
        self._model_name   = modelName
        self._bucket       = bucket

        @property
        def project_name(self):
            return self._project_name
        
        @property
        def model_name(self):
            return self._model_name

        @property
        def bucket(self):
            return self._bucket
        
    def local_tifs(self, name_selector:str='WSE', file_format:str='.tif', local_path:str='') -> list:
        """
            Search a local directory to find tiffs
        """
        assert local_path != '', "For local results, directory containing paths must be specified"
        return list(pl.PurePosixPath(local_path).glob("*.tif"))
            
    def s3_tifs(self, nameSelector:str='WSE', fileformat:str='.tif') -> list:
        """
        From function s3List, implemented here as a class method.
        """
        s3_client = boto3.client('s3')
        keys = s3_client.list_objects_v2(Bucket=self._bucket, Prefix=self._project_name)
        keysList = [keys]
        pathsList = []
        
        while keys['IsTruncated'] is True:
            keys = s3_client.list_objects_v2(Bucket=self._bucket, Prefix=self._project_name,
                                             ContinuationToken=keys['NextContinuationToken'])
            keysList.append(keys)
        for key in keysList:
            key_matches = [elem['Key'] for elem in key['Contents'] if elem['Key'].find('{}'.format(nameSelector))>=0 and elem['Key'].endswith(fileformat)]
            paths = ['s3://{0}/{1}'.format(self._bucket, key) for  key in key_matches]
            pathsList.extend(paths)

        return pathsList


class PFRAGrid:

    def __init__(self, tiff:str):
        self._tiff          = tiff
        self._posix_path    = pl.PurePosixPath(self._tiff)
        self._tiff_name     = self._posix_path.name
        
        def is_local(self):
            if self._posix_path.parts[0]=='s3:':
                return False
            else:
                return True
            
        self._is_local = is_local(self)
            
        def read_from_s3(self) -> 'gdal objects':
            assert not self._is_local, 'Tiff must be on s3 to use this function'
            s3Obj = s3.Object(self._bucket, self._prefix)
            image_data = BytesIO(s3Obj.get()['Body'].read())
            tif_inmem = "/vsimem/data.tif" #Virtual Folder to Store Data
            gdal.FileFromMemBuffer(tif_inmem, image_data.read())
            src = gdal.Open(tif_inmem)  
            return src.GetRasterBand(1), src.GetGeoTransform(), src
        
        def read_from_local(self) -> 'gdal objects':
            src = gdal.Open(self._tiff)  
            return src.GetRasterBand(1), src.GetGeoTransform(), src
            
        if self._is_local:
            self._bucket, self._prefix = None, None
            self._rasterBand, self._geoTrans, self._src = read_from_local(self)
            
        else:
            self._bucket = self._posix_path.parts[1]
            self._prefix = '/'.join(self._posix_path.parts[2:]) 
            self._rasterBand, self._geoTrans, self._src = read_from_s3(self)
            
    @property
    def posix_path(self):
        return self._posix_path

    @property
    def tiff_name(self):
        return self._tiff_name
    
    @property
    def rb(self):
        return self._rasterBand
    
    @property
    def gt(self):
        return self._geoTrans
    
    @property
    def src(self):
        return self._src
    
    @property
    def no_data_value(self):
        return self._rasterBand.GetNoDataValue()
    
    @property
    def projection_string(self):
        try:
            tiff_crs = osr.SpatialReference(self._src.GetProjectionRef()).ExportToProj4()
            tiff_crs = rasterio.crs.CRS.from_proj4(tiff_crs).to_proj4()
            return rasterio.crs.CRS.from_string(tiff_crs)
        except:
            print('Check Tiff Coordinate System, osr unable to find projection data')
            return None
    
    
class WSEGrid(PFRAGrid):

    def __init__(self, tiff):
        
        super().__init__(tiff)
        
        self._project_name  = self._tiff_name.split('_')[0]
        self._model_subtype = self._tiff_name.split('_')[1]
        self._event_id      = self._tiff_name.split('_')[2].replace('.tif','')
        
    @property
    def project_name(self):
        return self._project_name

    @property
    def model_subtype(self):
        return self._model_subtype        

    @property
    def event_id(self):
        return self._event_id
    
    
        
class TOPOGrid(PFRAGrid):
    
    def __init__(self, tiff):
        
        super().__init__(tiff)
        
        def get_metadata(self):
            pass
        
    @property
    def metadata(self):
        pass       
            

class StructurePoints:
    
    def __init__(self, shapefile:str) -> bool:
        
        self.required_fields = ['BLDG_DED', 'BLDG_LIMIT']
        self._shapefile  = shapefile
        self._shapefile_path  = pl.PurePosixPath(self._shapefile)
        
        def get_geodataframe(self):
            return gpd.read_file(self._shapefile)
        
        self.geodataframe = get_geodataframe(self)
        
        def check_fields(self):
            '''Check required fields'''
            missing_fields = [f for f in self.required_fields if f not in self.geodataframe.columns]
            assert len(missing_fields) < 1, "Required fields not found in shapefile: {}".format(str(missing_fields)) 
            
        self._field_check = check_fields(self)
        
        def get_projection(self):
            '''return projection'''
            return self.geodataframe.crs
            pass
        
        self._current_projection = get_projection(self)
        
    @property
    def projection_string(self):
        gdf_crs = rasterio.crs.CRS.from_dict(self._current_projection).to_proj4()
        return rasterio.crs.CRS.from_string(gdf_crs)
         
    
    

def mapper(fnc):
    '''Decorator for query_gdf to pass all points as a list'''
    def inner(gdf, gdf_index, gt, rb):
        noDataValue = rb.GetNoDataValue()
        return [fnc(gdf,idx, gt, rb) for idx in list(gdf_index)]
    return inner

@mapper
def query_gdf(gdf: gpd.geodataframe, idx:int,  gt: any, rb: any, point_id:str='ITEMID') -> dict:
    """
    Return point: pixel value pair for a given row in geodataframe
    """
    pointID = gdf[point_id].iloc[idx]
    x, y = gdf.iloc[idx].geometry.x, gdf.iloc[idx].geometry.y
    px = int((x-gt[0]) / gt[1])   
    py = int((y-gt[3]) / gt[5])   
    pixel_value = rb.ReadAsArray(px,py,1,1)[0][0]
    return {pointID:pixel_value}

# Placeholder to ensure topo & point datasets are of the same projection
def coordsCheck(vector_data, raster_data):
    if vector_data.projection_string == raster_data.projection_string:
        return True
    else:
        print("Check Projections:\n\tPoint data is in {}\n Gridded data is in {}".format(vector_data.projection_string,
                                                                                         raster_data.projection_string))
        return False