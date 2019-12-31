'''
PFRA Module for working with HEC-RAS model input files
'''
import os
import shutil
import pathlib as pl
import zipfile
import io
import datetime
import gzip
import json
import boto3
import h5py
import numpy as np
import pandas as pd

s3 = boto3.resource('s3')

class RasModel(object):
    '''
    This object holds information for the files stored in a hec-ras zip file used for STARRII PFRA study.
    If path starts with s3 then the code will run from s3 zipfile, otherwise path is expected
    to be a string path to a zipped local model (e.g. *.zip)
    '''
    def __init__(self, path:str, s3_data:bool=True, zipped:bool=True, verbose:bool=False):
        assert 'zip' in path, "Model files must be stored in a .zip file"

        def getS3Zip(self):
            '''Returns zipfile data from s3'''
            path_parts = pl.PurePosixPath(self.s3path).parts
            bucket = path_parts[1]
            key = '/'.join(path_parts[2:])
            obj = s3.Object(bucket_name=bucket, key=key)
            buffer = io.BytesIO(obj.get()["Body"].read())
            return zipfile.ZipFile(buffer)

        self.s3path      = path
        self.name        = str(pl.PurePosixPath(self.s3path).name).replace('.zip','')
        self._modelType  = self.name.split('_')[1][0]
        self._subType    = self.name.split('_')[2]
        self._zipfile    = getS3Zip(self)
        self._contents   = [x.filename for x in self._zipfile.infolist()]

        try:
            self.prj_file = [x for x in self._contents if '{}.prj'.format(self.name) in x][0]
        except:
            print('No prj file found')

    @property
    def zipfile(self):
        return self._zipfile

    @property
    def subType(self):
        return self._subType

    @property
    def modelType(self):
        modelType  = self._modelType
        assert modelType =='F' or modelType == 'P', 'Check Model Nomenclature, expected a P or F Model, found {}'.format(modelType)
        if self._modelType =='F':
            return 'Fluvial'

        elif self._modelType =='P':
            return 'Pluvial'

    @property
    def contents(self):
        return self._contents


class RasProject(object):
    '''Retrieve info on HEC-RAS Project Data from .prj file: designed principally to get plan data.
       Creates as a Class object so that new methods can be easily added to provide info on other
       parameters of interest in .prj file as needed.
    '''
    def __init__(self, model:RasModel):

        def read_lines(self):
            '''Read the prj file to a list separated by line breaks'''
            return self.zip_file.read(self.prj_file).decode('utf-8').splitlines()

        def get_current_plan_ext(self):
            '''Returns the current plan extension in the prj'''
            for line in self.prj_data:
                if 'Current Plan' in line:
                    extension = line.split('=')[1].strip('\n')

                    return extension

        def get_file_list(self, file_type):
            '''Returns a list of files by type'''
            file_list=[]

            for line in self.prj_data:
                if file_type in line:
                    file_list.append(line.split("=")[1])

            return file_list

        self.model             = model
        self.prj_file          = self.model.prj_file
        self.zip_file          = self.model.zipfile
        self.prj_data          = read_lines(self)
        self._current_plan_ext = get_current_plan_ext(self)
        self._geometry_files   = get_file_list(self, "Geom File")
        self._unsteady_files   = get_file_list(self, "Unsteady File")
        self._plan_files       = get_file_list(self, "Plan File")

    @property
    def current_plan(self):
        return self._current_plan_ext

    @property
    def geom_files(self):
        return self._geometry_files

    @property
    def unsteady_files(self):
        return self._unsteady_files

    @property
    def plan_files(self):
        return self._plan_files


class RasPlan():
    '''Retrieve info on HEC-RAS Project Data from .prj file: designed principally to get plan data.
       Creates as a Class object so that new methods can be easily added to provide info on other
       parameters of interest in .prj file as needed.
    '''
    def __init__(self, model:RasModel, current_plan:str):

        def plan_path(self):
            try:
                plan_path_name = ''.join([self.model.name, '.' , current_plan])
                return [x for x in self.model.contents if x == plan_path_name][0]
            except IndexError as e:
                print('{} Not Found in zip, {}'.format(plan_path_name, e))

        def read_lines(self):
            '''Read the prj file to a list separated by line breaks'''

            return self.model.zipfile.read(self.plan_path).decode('utf-8').splitlines()

        def get_data(self, data):
            '''Returns data by keyword search'''
            data_list=[]

            for line in self.plan_data:
                if data in line:
                    data_list.append(line.split("=")[1])

            return data_list

        self.model                = model
        self.plan_path            = plan_path(self)
        self.plan_data            = read_lines(self)
        self._hdfFilePath         = self.plan_path +'.hdf'
        self._geometry_files      = get_data(self, "Geom File")
        self._flow_files          = get_data(self, "Flow File")
        self._SimulationDate      = get_data(self, "Simulation Date")
        self._ComputationInterval = get_data(self, "Computation Interval")
        self._OutputInterval      = get_data(self, "Output Interval")
        self._ProgramVersion      = get_data(self, "Program Version")
        self._MappingInterval     = get_data(self, "Mapping Interval")
        self._PlanTitle           = get_data(self, "Plan Title")
        self._ShortIdentifier     = get_data(self, "Short Identifier")

    @property
    def geom_files(self):
        return self._geometry_files

    @property
    def unsteady_files(self):
        return self._flow_files

    @property
    def simulationDate(self):
        return self._SimulationDate[0].strip()

    @property
    def computationInterval(self):
        return self._ComputationInterval[0].strip()

    @property
    def outputInterval(self):
        return self._OutputInterval[0].strip()

    @property
    def programVersion(self):
        assert self._ProgramVersion[0] == '5.07' ,'Model Version Incorrect, Model must be created and tested on version 5.0.7'
        return self._ProgramVersion[0]

    @property
    def planTitle(self):
        return self._PlanTitle[0].strip()

    @property
    def shortIdentifier(self):
        return self._ShortIdentifier[0].strip()

    @property
    def hdfFilePath(self):
        return self._hdfFilePath


class HDFPlanFile(object):
    '''
    Returns an open r+ copy of HEC-RAS Unsteady HDF Plan File
    '''
    def __init__(self, model:RasModel, planData:RasPlan):

        def copy_to_local(self):
            self.model.zipfile.extract(self._zip_path)
            return h5py.File(self._zip_path, 'r+')

        def get_2dFlowArea_data(self):
            h5_path =  '/Geometry/2D Flow Areas/Attributes'
            table_data = np.array(self._hdfLocal[h5_path].value)
            table = pd.DataFrame.from_records(table_data, columns=list(table_data.dtype.fields.keys()))
            table['Name'] = table['Name'].str.decode("utf-8")
            return table

        def get_planHydrographs(self):
            # initalize dict to store hydrographs
            graphs={}
            event_data = 'Event Conditions/Unsteady/Boundary Conditions'

            # event_data groups (e.g. 'Flow Hydrographs', 'Precipitation Hydrographs', etc)
            for forcingType in self._hdfLocal[event_data].keys():
                # forcingData (e.g. 'F01, D01')
                for forcingData in self._hdfLocal['{}/{}'.format(event_data, forcingType)].keys():
                    hydro_data_address = '{}/{}/{}'.format(event_data, forcingType, forcingData)
                    hydro_data = self._hdfLocal[hydro_data_address]
                    graphs[hydro_data_address] = np.asarray(hydro_data)

            return graphs

        def get_mandatory_files(self):
            '''Use extensions to identify files used by linux unsteady'''
            h5_path =  '/Plan Data/Plan Information'
            table_data = dict(self._hdfLocal[h5_path].attrs.items())

            geom_exts = [line.split('=')[1] for line in planData.plan_data if 'Geom File' in line]
            assert len(geom_exts)==1, 'Expected 1 geometry file, found several'
            plan_ext = pl.Path(planData.plan_path).suffix

            mandatory_files = [model.name + plan_ext.replace('.p', '.b'),
                               model.name + geom_exts[0].replace('g', '.x'),
                               model.name + geom_exts[0].replace('g', '.c')]

            #mandatory_files = [table_data['Plan Filename'].decode().replace('.p', '.b'),
            #                   table_data['Geometry Filename'].decode().replace('.g', '.x'),
            #                   table_data['Geometry Filename'].decode().replace('.g', '.c')]




            return mandatory_files

        def get_restart_files(self):
            rstFiles = [f for f in self.model.contents if f.split('.')[-1] == 'rst']
            return rstFiles

        self.model                 = model
        self._zip_path             = planData.hdfFilePath
        self._hdfLocal             = copy_to_local(self)
        self._2dFlowArea           = get_2dFlowArea_data(self)
        self._planHydrographData   = get_planHydrographs(self)
        self._planHydrographsList  = list(self._planHydrographData.keys())
        self._domains              = self._2dFlowArea['Name'].values
        self._cellCount            = self._2dFlowArea['Cell Count'].sum()
        self.mandatoryFiles       = get_mandatory_files(self)
        self._restartFiles         = get_restart_files(self)

    @property
    def restartFiles(self):
        return self._restartFiles

    @property
    def hdfLocal(self):
        return self._hdfLocal

    #@property
    #def mandatoryFiles(self):
    #    return self.mandatoryFiles

    @property
    def get_2dFlowArea(self):
        return self._2dFlowArea

    @property
    def planHydrographData(self):
        return self._planHydrographData

    @property
    def planHydrograpList(self):
        return self._planHydrographsList

    @property
    def domains(self):
        return self._domains

    @property
    def cellCount(self):
        return self._cellCount

    def updateSimDates(self, eventID: str, start_date:str, end_date:str):

        for bcName in self.planHydrographData.keys():
            bcData = self.hdfLocal[bcName]
            for name, data in bcData.attrs.items():

                if 'Start Date' in name:
                    bcData.attrs.modify('Start Date', bytes(start_date, 'utf-8'))

                elif 'End Date' in name:
                    bcData.attrs.modify('End Date', bytes(end_date, 'utf-8'))

        # Update sim start date in plan info
        simStartDate = datetime.datetime.strptime(start_date, "%d%b%Y %H%M").strftime("%d%b%Y %H:%M:%S")
        simEndDate = datetime.datetime.strptime(end_date, "%d%b%Y %H%M").strftime("%d%b%Y %H:%M:%S")
        planInfo = '/Plan Data/Plan Information'
        self.hdfLocal[planInfo].attrs.modify('Simulation Start Time', bytes(simStartDate, 'utf-8'))
        self.hdfLocal[planInfo].attrs.modify('Simulation End Time', bytes(simEndDate, 'utf-8'))
        self.hdfLocal[planInfo].attrs.modify('Time Window', bytes(' to '.join([simStartDate, simEndDate]), 'utf-8'))
        self.hdfLocal[planInfo].attrs.modify('Flow Title', bytes(eventID, 'utf-8'))


    def updateHydrograph(self, hydro_key:str, eventData:np.array):
        '''Create a temporary dataset with updated event data, copy attrs of current boundary condition
           Delete current dataset, copy temp, delete temp.
           eventData is time series forcing array (time, value)
           Room for improvement
        '''

        bcName       = str(pl.PurePosixPath(hydro_key).name)
        current_data = self.hdfLocal[hydro_key]

        # Write Temporary data
        tmp = str(pl.PurePosixPath(hydro_key).parent/'tmp')
        self.hdfLocal.create_dataset(tmp, shape=eventData.shape, dtype=current_data.dtype, data=eventData)

        for name, data in current_data.attrs.items():
            attr = current_data.attrs[name]
            self.hdfLocal[tmp].attrs.create(name, data, shape=None, dtype=None)

        # Delete Current data
        del self.hdfLocal[hydro_key]

        # Copy Temporary data to Current data namespace
        self.hdfLocal.create_dataset(hydro_key, shape=eventData.shape, dtype=eventData.dtype, data=eventData)

        for name, data in current_data.attrs.items():
            attr = current_data.attrs[name]
            self.hdfLocal[hydro_key].attrs.create(name, data, shape=None, dtype=None)


        # Delete Temp data
        del self.hdfLocal[tmp]


    def saveHDF(self, newPath):
        eventHDF = h5py.File(newPath, 'w')
        for group in self.hdfLocal.keys():
            if group not in["Results"]:
                self.hdfLocal.copy(group, eventHDF)

        assert eventHDF != None, 'Error saving  HDF file {}'.format(newPath)

        if eventHDF != None:
            eventHDF.close()

        return newPath

class RASForcing(object):
    def __init__(self, path:str, s3_data:bool=True, zipped:bool=True, verbose:bool=False):
        assert 'zip' in path, "Model files must be stored in a .zip file"

        def getS3Zip(self):
            '''Returns zipfile data from s3'''
            path_parts = pl.PurePosixPath(self.s3path).parts
            bucket = path_parts[1]
            key = '/'.join(path_parts[2:])
            obj = s3.Object(bucket_name=bucket, key=key)
            buffer = io.BytesIO(obj.get()["Body"].read())

            return zipfile.ZipFile(buffer)

        def getDomainForcing(self):
            domainData =  [(f.split('_')[-1].replace('.json',''),f) for f in self.contents]
            forcingData = {}
            for domain, domainFile in domainData:
                json_domain_data = self._zipfile.read(domainFile)
                forcingData[domain] = json.loads(json_domain_data)

            return forcingData

        def getProductionEventsFromJSON(self):
            events=[]
            for domainName, domainData in self._domainForcing.items():
                # modelDescriptor could be 'Fluvial' or 'H06', 'H12'....
                for modelDescriptor in domainData.keys():
                    forcingData = self._domainForcing[domainName][modelDescriptor]

                    for bcName, bcData in forcingData['BCName'].items():
                        for eventID, eventData in forcingData['BCName'][bcName].items():
                            events.append(eventID)

            return list(set(events))

        self.s3path             = path
        self.name               = str(pl.PurePosixPath(self.s3path).name).replace('.zip','')
        self._zipfile           = getS3Zip(self)
        self._contents          = [x.filename for x in self._zipfile.infolist()]
        self._domainForcing     = getDomainForcing(self)
        self._productionEvents  = getProductionEventsFromJSON(self)

    @property
    def zipfile(self):
        return self._zipfile

    @property
    def contents(self):
        return self._contents

    @property
    def domainForcing(self):
        return self._domainForcing

    @property
    def productionEvents(self):
        return self._productionEvents



#--------------------------------FUNCTIONS-------------------------------------#

def checkS3ForFile(path:str):
    '''Search s3 for event outputs
       Needs to be revisited
    '''
    bucket, key = splitS3Path(path.replace('\\','/'))
    s3_obj = s3.Object(bucket, key)
    spath, filename = os.path.split(s3_obj.key)
    my_bucket = s3.Bucket(bucket)
    objs = list(my_bucket.objects.filter(Prefix=key))
    if len(objs) > 0 and objs[0].key == key:
        return True
    else:
        return False


def checkIfS3EventsExist(events:list, projectName:str, subType:str):
    '''Search s3 for events to prevent overwriting
       Needs to be revisited
    '''
    intervals = ['H06', 'H12', 'H24', 'H96']

    missingScalingEvents = []
    for event in events:
        eventExistsIns3 = False
        for interval in intervals:
            zipName = '{}_{}_{}_{}_in'.format(projectName, subType, interval, event)
            s3Path = 's3://pfra/{}/{}/{}/{}/{}.zip'.format(projectName, subType, interval, event, zipName)
            check = checkS3ForFile(s3Path)
            if check == True:
                eventExistsIns3 =True
                break;
        if eventExistsIns3 == False:
            missingScalingEvents.append(event)

    if len(missingScalingEvents) > 0:
        return False
    else:
        return True

def scaleTestEvents(path):
    '''Specified runs for PFRA scale tests'''
    scale_events = {'Pluvial': ["E0001", "E0101", "E1001", "E1101",
                               "E2001", "E2101", "E3001", "E3101"],
                    'Fluvial': ["E0001", "E0025", "E0075", "E0100"]}

    if 'P' in pl.Path(path).name.split('_')[1]:
        return scale_events['Pluvial']
    elif 'F' in pl.Path(path).name.split('_')[1]:
        return scale_events['Fluvial']
    else:
        print('Error in Path, unable to detect run type')
        return None

def GetModelEventData(e:str, forcing_data:dict):
    '''Given an event, search forcing zip to find all hydrographs for all domains for all boundary conditions'''
    runData={}
    for domain in forcing_data.domainForcing.keys():
        runData[domain]={}
        for modelDescriptor, domainData in forcing_data.domainForcing[domain].items():
            #print(modelDescriptor)
            for bc in domainData['BCName'].keys():

                if e in domainData['BCName'][bc].keys():
                    time_idx    = domainData['time_idx']
                    run_duration_days = domainData['run_duration_days']

                    start_dtm   = datetime.datetime.strptime("01May2000 1000", "%d%b%Y %H%M")
                    simDuration = datetime.timedelta(days=run_duration_days)
                    end_dtm     = start_dtm + simDuration

                    start_date = start_dtm.strftime("%d%b%Y %H%M")
                    end_date   = end_dtm.strftime("%d%b%Y %H%M")

                    bcTimeSeries = domainData['BCName'][bc][e]
                    bcTimeIDX = np.divide(time_idx,24)

                    runData[domain][bc] = {'modelDescriptor':modelDescriptor,
                                           'start_date':start_date,
                                           'end_date' :end_date,
                                           'tseries' : np.array(list(zip(bcTimeIDX ,bcTimeSeries)),dtype=np.float32)}
    return runData

def bc_hdf_path(model:str, domain_name:str, bc_name:str, plan_hydrographs:list):
    '''Given domain name, bc name, and list of bcs in model, 
       create lust of full paths to write forcing data to hdf
       
       NOTES
       For Pluvial Forcing, precip bc names are like 'SA: D01'
       All other boundaries (Fulvial or Pluvial models ) are like: SA: D01 BCLine: P01_D01_L02'
    '''
    
    # strip the bc name from the full path for all bcs in HDF
    bcNames = [str(pl.Path(f).name) for f in plan_hydrographs]
    # 'Event Conditions/Unsteady/Boundary Conditions/Flow Hydrographs/SA: D01 BCLine: P01_D01_L02' -> 'SA: D01 BCLine: P01_D01_L02'
    if bc_name == domain_name:
        bc_name_lookup = 'SA: {}'.format(bc_name)
    else:
        bc_name_lookup = 'SA: {} BCLine: {}'.format(domain_name, bc_name)
        bc_name_lookup_with_nxline = 'SA: {} BCLine: \n{}'.format(domain_name, bc_name)
        bc_name_lookup_with_2nxline = 'SA: {} BCLine:  \n{}'.format(domain_name, bc_name)

    hdf_bc_path = [f for f in plan_hydrographs if str(pl.Path(f).name)==bc_name_lookup]
    
    if len(hdf_bc_path)==0:
        hdf_bc_path = [f for f in plan_hydrographs if str(pl.Path(f).name)==bc_name_lookup_with_nxline]
        
        if len(hdf_bc_path)==0:
            hdf_bc_path = [f for f in plan_hydrographs if str(pl.Path(f).name)==bc_name_lookup_with_2nxline]

    assert len(hdf_bc_path) == 1, 'Unable to Find BC for {} in {}'.format(domain_name, bc_name)

    return hdf_bc_path[0]





def prepEventZipFiles(eventID:str, model:RasModel, localPlan:HDFPlanFile, prjData:RasProject, modelDescriptor:str ,
                     processing_dir = pl.PurePosixPath(os.getcwd())):
    '''Make Local copies of files to zip up for simulation
       hdf plan file is ready, other files need to be overwritten

    '''
    filesToProcess={}
    mandatory_files=[]

    # Correct paths if zip contents are in a subdirectory (e.g. atkinstest/atkinstest_P01_H00.prj)
    if len(pl.Path(prjData.prj_file).parts) > 1:
         modelFolder = pl.Path(prjData.prj_file).parts[0]
    else:
        modelFolder=''

    # Check for restart File
    if len(localPlan.restartFiles) > 0:
        assert len(localPlan.restartFiles) <= 1, 'Too many restart files included in zipfile'
        if len(localPlan.restartFiles) ==1:
            restartFile = str(pl.Path(localPlan.restartFiles[0]).name)
            restart_file_found = True
    else:
        restart_file_found = False


    if restart_file_found:
        mandatory_files = localPlan.mandatoryFiles
        mandatory_files.append(restartFile)
    else:
        mandatory_files = localPlan.mandatoryFiles

    zipFileNames = [f for f in model.contents if str(pl.Path(f).name) in mandatory_files]
    eventComputeFiles = [model.zipfile.extract(f, processing_dir) for f in zipFileNames]
    
    if model.modelType =='Pluvial':
        for f in eventComputeFiles:
            name = pl.Path(f).name
            newName = name.replace('H00', modelDescriptor)
            fNameWithDuration = str(pl.Path(f).name.replace(name, newName)).replace(modelDescriptor, '{}_{}'.format(modelDescriptor,eventID))
            # Don't copy the Restart file because the name is the same (name doesn't need to change)
            if '.rst' not in f: 
                shutil.copy(f, fNameWithDuration)
                eventComputeFiles[eventComputeFiles.index(f)]=fNameWithDuration
                os.remove(f)

    if model.modelType =='Pluvial':
        eventHDFName = processing_dir/modelFolder/'{}_{}.{}.tmp.hdf'.format(model.name.replace('H00', modelDescriptor),eventID,
                                                                     prjData.current_plan )

    elif model.modelType == 'Fluvial':
        if modelFolder:
            eventHDFName =  processing_dir/modelFolder/'{}.{}.tmp.hdf'.format(model.name,
                                                                      prjData.current_plan)
        else:  
            eventHDFName =  processing_dir/'{}.{}.tmp.hdf'.format(model.name,
                                                                      prjData.current_plan)

    localPlan.saveHDF(eventHDFName)
    eventComputeFiles.append(eventHDFName)

    for computeFile in eventComputeFiles:
        
        fname  = pl.Path(computeFile).name
        if '.rst' in fname:
            filesToProcess[fname] = computeFile
        else:
            runName = fname.replace(model.subType, '{}_{}'.format(model.subType, eventID))
            filesToProcess[runName] = computeFile

    filesToProcess['zipName'] = str(processing_dir/modelFolder/runName.split('.')[0])+'_in.zip'

    return filesToProcess

def updateBfile(computeFile:str, model:RasModel, eventID:str,start_date:str, end_date:str):

    with open(computeFile, 'r') as f:
        lines = f.readlines()

    idx = lines.index("Project Title, Plan Title and Plan ShortID\n")
    lines[idx+1] = '{}\n'.format(model.name)
    lines[idx+2] = '{}\n'.format(eventID)
    lines[idx+3] = '{}\n'.format(eventID)

    idx = lines.index("Computational Time Window\n")
    lines[idx+1] = lines[idx+1].replace(lines[idx+1].split('= ')[1],'{}\n'.format(start_date))
    lines[idx+2] = lines[idx+2].replace(lines[idx+2].split('= ')[1],'{}\n'.format(end_date))

    # Add line for restart file...
    #idx = lines.index("Initial Conditions (use restart file?)\n")
    #lines[idx+1] = lines[idx+1].replace(lines[idx+1].split('= ')[1],'{}\n'.format(start_date))

    with open(computeFile, 'w') as f:
        for line in lines:
            f.write(line)

def updateXfile(computeFile:str, eventID:str, start_date:str, end_date:str):

    with open(computeFile, 'r') as f:
        lines = f.readlines()

    idx = lines.index("Section - Arrays Sizes\n")
    lines[idx+1] = '{}\n'.format(eventID)

    with open(computeFile, 'w') as f:
        for line in lines:
            f.write(line)

def updateComputeFiles(model:RasModel, rawModelFiles:dict, eventID:str, start_date, end_date):
    '''This should be moved to a method within one of the class objects to remove nesting'''
    for fname, f in rawModelFiles.items():
        if 'b' in pl.Path(f).suffix:
            updateBfile(f, model, eventID, start_date, end_date)

        elif 'x' in pl.Path(f).suffix:
            updateXfile(f, eventID, start_date, end_date)

def pushModeltoS3(rawModelFiles:dict, bucket:str='pfra'):
    '''Build s3 path, zip files, garbage collection'''
    
    eventZipLocalPath = rawModelFiles['zipName']
    
    runName = pl.PurePosixPath(eventZipLocalPath).name
    runDescriptors = runName.split('_')

    projectArea    = runDescriptors[0]
    modelName      = runDescriptors[1]
    modelSubtype   = runDescriptors[2]
    eventID        = runDescriptors[3]

    s3_prefix = "{0}/{1}/{2}/{3}/{4}".format(projectArea, modelName, modelSubtype, eventID, runName)
    
    with zipfile.ZipFile('{}'.format(eventZipLocalPath) , mode='w') as zf:
        for fname, f in rawModelFiles.items():
            if fname != 'zipName':
                zf.write(f, fname)
                
    s3.Bucket(bucket).upload_file(eventZipLocalPath, s3_prefix)
    
    #Garbage Collection
    for fname, f in rawModelFiles.items():
        os.remove(f)

    return 's3://{}/{}'.format(bucket, s3_prefix)