# Functions used in the PrepWSEs Notebook
#
# Author: Alec Brazeau
# abrazeau@dewberry.com

# imports
from time import time
import zipfile
from s3utils import load_s3_csv, load_s3_json
import io
import os
import geopandas as gpd
from io import BytesIO
import numpy as np
import gdal
import boto3
from glob import glob
import osr
import pandas as pd

def get_s3_zipped_terrain(bucket:str, key:str):
    """Get the terrain tif name and its data from an s3 zip"""
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name=bucket, key=key)
    buffer = io.BytesIO(obj.get()["Body"].read())
    data = zipfile.ZipFile(buffer)
    contents = data.namelist()
    tifs = [t for t in contents if '.tif' in t and not 'Mannings_n' in t and not 'Land' in t and 'landuse' not in t]
    print(tifs)
    assert len(tifs) == 1, 'Too many tifs, expected 1, got {}!'.format(len(tifs))
    return tifs[0], data


def read_zipped_tif(tif_name:str, zip_data:any):
    """Read a zipped tif into memory using gdal"""
    tif_inmem = "/vsimem/data.tif" #Virtual Folder to Store Data
    gdal.FileFromMemBuffer(tif_inmem, zip_data.open(tif_name).read())
    src = gdal.Open(tif_inmem)  
    return src.GetRasterBand(1), src.GetGeoTransform(), src


def get_terrain(bucket:str, key:str):
    """Wrap the `get_s3_zipped_terrain` function and the `read_zipped_tif`"""
    return read_zipped_tif(*get_s3_zipped_terrain(bucket, key))


def query_gdf(gdf: gpd.geodataframe, gt: any, rb: any, point_id:str) -> dict:
    """Return point: pixel value pair for a given row in geodataframe"""
    results={}
    for idx in gdf.index:
        pointID = gdf[point_id].iloc[idx]
        x, y = gdf.iloc[idx].geometry.x, gdf.iloc[idx].geometry.y
        px = int((x-gt[0]) / gt[1])   
        py = int((y-gt[3]) / gt[5])  
        try:
            pixel_value = rb.ReadAsArray(px,py,1,1)[0][0]
            results[pointID] = pixel_value
        except TypeError as e:
            results[pointID] = 'Error, verify projection is correct and Point is within Tiff bounds'      
    return results


def make_WSE_df(s3files: list, print_mod:int = 100):
    """Merge wse dataframes when given a list of them on s3"""
    st = time()
    dfs = []
    for i, f in enumerate(s3files):
        project, model, subtype, event = parse_WSE_path(f)
        df = load_s3_csv(f)
        df = df.rename(columns={'Unnamed: 0': 'plus_code', os.path.basename(f).split('.')[0]: subtype + '_' + event})
        df = df.set_index('plus_code')
        dfs.append(df)
        if i % print_mod == 0:
            print("{} Events Processed".format(i))
    results = pd.concat(dfs, axis=1)
    print(round((time() - st)/60, 2), 'minutes to run')
    return results.replace(-9999.0, np.nan)


def parse_WSE_path(s3path: str):
    """Parse the WSE path to get the event id info"""
    bn = os.path.basename(s3path)
    parts = bn.split('.')[0].split('_')
    project = parts[0]
    model = parts[1]
    subtype = parts[2]
    event = parts[3]
    return project, model, subtype, event


def get_elevations(project:str, model:str, points_shapefile:str):
    """Pull elevations from the terrain tif file given a set of points"""
    st = time()
    if 'F' in model:
        key = '{0}/BaseModels/{0}_{1}_NBR.zip'.format(project, model)
    else:
        key = '{0}/BaseModels/{0}_{1}_H00.zip'.format(project, model)
    bucket  = 'pfra'
    rb, gt, src = get_terrain(bucket, key)
    proj4 = osr.SpatialReference(src.GetProjectionRef()).ExportToProj4()
    gdf = gpd.read_file(points_shapefile).to_crs(proj4)
    assert 'plus_code' in gdf.columns, "Check Point file, no plus_code column found"
    
    pt_dict = query_gdf(gdf, gt, rb, 'plus_code')
    elev = pd.DataFrame([pt_dict]).T.reset_index().rename(columns={'index':'plus_code', 0:'GroundElev'})
    gdf = pd.merge(gdf, elev, on='plus_code')
    gdf = gdf[['plus_code', 'GroundElev']]
    print(round((time() - st)/60, 2), 'minutes to run')
    return gdf.set_index('plus_code')


def parse_weights_path(s3path: str):
    bn = os.path.basename(s3path)
    parts = bn.split('.')[0].split('_')
    project = parts[0]
    model = parts[1]
    domain = parts[2]
    return project, model, domain


def make_weights_df(s3path: str):
    subtype_dict = {'0' : 'H06', '1' : 'H12', '2' : 'H24', '3' : 'H96'}
    project, model, domain = parse_weights_path(s3path)
    try:
        wt = load_s3_json(s3path)
        weights = pd.DataFrame()
        weights['event'] = list(wt['BCName'][domain].keys())
        weights['weight'] = list(wt['BCName'][domain].values())
        weights['project'] = project
        weights['model_type'] = model
        weights['model_sub_type'] = weights.event.apply(lambda x: subtype_dict[x[1]])
        weights['event_id'] = weights.model_sub_type + '_' + weights.event
        weights['job_id'] = weights.event_id.apply(lambda x: r's3://pfra/{0}/{1}/{2}/{3}/{4}_in.zip'.format(
            project,
            model,
            x.split('_')[0],
            x.split('_')[1],
            '_'.join([project, model, x.split('_')[0], x.split('_')[1]])))
        return weights.set_index('job_id')
    
    except TypeError as e:
        print('Check Input Path and rerun')
