import boto3
import os
from multiprocessing import Pool, cpu_count
import pandas as pd
import json


def s3List(bucketName, prefixName, nameSelector, fileformat):
    """
        This function takes an S3 bucket name and prefix (flat directory path) and returns a list of netcdf file.
            This function utilizes boto3's continuation token to iterate over an unlimited number of records.

        BUCKETNAME -- A bucket on S3 containing GeoTiffs of interest
        PREFIXNAME -- A S3 prefix.
        NAMESELECTOR -- A string used for selecting specific files. E.g. 'SC' for SC_R_001.tif.
        FILEFORMAT -- A string variant of a file format.
    """
    # Set the Boto3 client
    s3_client = boto3.client('s3')
    # Get a list of objects (keys) within a specific bucket and prefix on S3
    keys = s3_client.list_objects_v2(Bucket=bucketName, Prefix=prefixName)
    # Store keys in a list
    keysList = [keys]
    # While the boto3 returned objects contains a value of true for 'IsTruncated'
    while keys['IsTruncated'] is True:
        # Append to the list of keys
        # Note that this is a repeat of the above line with a contuation token
        keys = s3_client.list_objects_v2(Bucket=bucketName, Prefix=prefixName,
                                         ContinuationToken=keys['NextContinuationToken'])
        keysList.append(keys)
    # Create a list of GeoTiffs from the supplied keys
    #     While tif is hardcoded now, this could be easily changed.
    pathsList = []
    for key in keysList:
        paths = ['s3://' + bucketName + '/' + elem['Key'] for elem in key['Contents'] \
                 if elem['Key'].find('{}'.format(nameSelector)) >= 0 and elem['Key'].endswith(fileformat)]
        pathsList = pathsList + paths
    return pathsList


def s3download(s3paths: list, download_dir: str):
    """Downloads all s3 files in a given list of s3paths to a specified directory"""
    s3client = boto3.client("s3")
    for f in s3paths:
        bucket = f.split(r"s3://")[1].split(r"/")[0]
        key = f.split(f's3://{bucket}/')[-1]
        filename = os.path.basename(f)
        if not os.path.exists(os.path.join(download_dir, filename)):
            s3client.download_file(bucket, key, os.path.join(download_dir, filename))


def _single_dl(args: tuple):
    """Unpack tuple of arguments (s3path, download directory) to allow parallel download from s3"""
    file = args[0]
    download_dir = args[1]
    s3client = boto3.client("s3")
    bucket = file.split(r"s3://")[1].split(r"/")[0]
    key = file.split(f's3://{bucket}/')[-1]
    filename = os.path.basename(file)
    if not os.path.exists(os.path.join(download_dir, filename)):
        s3client.download_file(bucket, key, os.path.join(download_dir, filename))


def s3download_parallel(s3paths: list, download_dir: str):
    """Downloads a all s3 files in a given list of s3paths to a specified directory in parallel (num of cores * 2)"""
    args = [(f, download_dir) for f in s3paths]
    p = Pool(int(cpu_count()*2))
    p.map(_single_dl, args)
    p.close()


def load_s3_json(s3path: str):
    """Load a json stored on s3 into memory"""
    s3 = boto3.resource('s3')
    bucket = s3path.split(r"s3://")[1].split(r"/")[0]
    key = s3path.split(r"{}/".format(bucket))[1]
    try:
        obj = s3.Object(bucket, key)
        content = obj.get()['Body'].read().decode('utf-8')
        return json.loads(content)
    
    except Exception as e:
        print('{} Does not exist, check input \n{}'.format(s3path, e))
        return None
    

def load_s3_csv(s3path: str):
    """Load a csv stored on s3 into memory"""
    s3 = boto3.client('s3')
    bucket = s3path.split(r"s3://")[1].split(r"/")[0]
    key = s3path.split(r"{}/".format(bucket))[1]
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])
    
    except Exception as e:
        print('{} Does not exist, check input {}'.format(s3path, e))
        return None
