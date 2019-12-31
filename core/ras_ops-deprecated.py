'''
PFRA Module for working with HEC-RAS model input files
'''
import os
import pathlib as pl
import sys
from glob import glob
import h5py
import numpy as np
import time
import json
import boto3
import zipfile
import io
import gzip
import pandas as pd
import matplotlib.pyplot as plt
s3 = boto3.resource('s3')

#-----------------------------------CLASSES------------------------------------#
class RasProject(object):
    '''Retrieve info on HEC-RAS Project Data from .prj file: designed principally to get plan data.
       Creates as a Class object so that new methods can be easily added to provide info on other
       parameters of interest in .prj file as needed.
    '''
    def __init__(self, path, project_name, verbose=False):
        self.errors=[]
        self.project_name = str(path)
        self.path = path
        self.verbose = verbose
        self.project_name = project_name

        try:
            #Open the .prj File.
            #If it is a zip file then search the zip for the correctly named file.
            #File contents are read into an array (self.lines).
            #An error will be recorded if there is not a correctly named .prj in the zip file
            if type(path) == zipfile.ZipFile:
                prjs = searchZip(path, ['.prj'])

                if verbose: [print(p) for p in prjs] # Debug

                ras_prj = [p for p in prjs if self.project_name in str(p)]

                if len(ras_prj) != 1:
                    self.errors.append(ErrorClass("Ras Project Not Found in zip {}".format(path.filename)))
                    return;
                else:
                    self.lines = path.read(ras_prj[0]).decode('utf-8').splitlines()
                    self.path = ras_prj[0].filename

            else:
                self.lines = read_lines(self.path)

        except:
            self.errors.append(ErrorClass("Error reading prj file {}".format(path)))
            return;

        #returns the current plan extension in the prj
        def get_current_plan_ext(self):
            extension = None
            for line in self.lines:
                if 'Current Plan' in line:
                    extension = line.split('=')[1].strip('\n')
                    return extension

        #returns the current plan in the prj
        def get_current_plan(self):
            extension = None
            try:
                for line in self.lines:
                    if 'Current Plan' in line:
                        extension = line.split('=')[1].strip('\n')
                if extension != None :
                    return self.path.replace('prj',extension)
                else:
                    #self.errors.append(ErrorClass("No Current Plan in prj file {}".format(path)))
                    return None
            except:
                self.errors.append(ErrorClass("No Current Plan in prj file {}".format(path)))

        #returns a list of all plan files in the prj
        def list_plans(self):
            plans = []
            for line in self.lines:
                if 'Plan File' in line:
                    plans.append(line.split('=')[1].strip('\n'))
            return plans

        #returns a list of all flow files in the prj
        def list_flows(self):
            flows = []
            for line in self.lines:
                if 'Flow File' in line:
                    flows.append(line.split('=')[1].strip('\n'))
            return flows

        #returns a list of all unsteady files in the prj
        def list_unsteadies(self):
            v = []
            for line in self.lines:
                if 'Unsteady File' in line:
                    v.append(line.split('=')[1].strip('\n'))
            return v

        #returns a list of all geom files in the prj
        def list_geom(self):
            v = []
            for line in self.lines:
                if 'Geom File' in line:
                    v.append(line.split('=')[1].strip('\n'))
            return v

        # Call init methods
        self.current_plan = (get_current_plan(self))
        self.current_plan_ext = (get_current_plan_ext(self))

        #If no errors in reading the prj, then set the rest of the prj properties
        if len(self.errors) == 0:
            self.list_plans = list_plans(self)
            self.list_flows = list_flows(self)
            self.list_unsteadies = list_unsteadies(self)
            self.list_geom = list_geom(self)
    #checks the RasProject for correctness. Returns a boolean and a list.
    def scaleTest(self):
        print('Initializing Scale Test')
        if self.current_plan == None:
            self.errors.append(ErrorClass("No Current Plan in prj file {}".format(self.path)))

        if len(self.list_geom) > 1:
            self.errors.append(ErrorClass("Too many Geom File listed in the prj file"))

        if len(self.list_unsteadies) > 1:
            self.errors.append(ErrorClass("Too many Unsteady File lines listed in the prj file "))


        if len(self.list_plans) > 1:
            self.errors.append(ErrorClass("Too many Plan File lines listed in the prj file"))


        if len(self.errors)==0:
            print('Scale Test Complete, Success!')
            return True, list()
        else:
            print('Scale Test Complete, Errors')
            return False, self.errors


class RasPlan(object):
    '''
       HEC-RAS Plan Data
       Add other items as needed to search_for dict

       ***This object is not currently used. UnsteadyHDFFile is used for plan information. ***
    '''
    def __init__(self, path):
        self.path = path
        self.path_extension = path.split('.')[-1]
        self.project_name = os.path.basename(path).split('.')[0]
        self.__search_for = {'geom_file':'Geom File', 'flow_file':'Flow File', 'sim_date':'Simulation Date',
                           'comp_int':'Computation Interval', 'out_int':'Output Interval', 'version':'Program Version',
                           'map_int':'Mapping Interval', 'title':'Plan Title', 'short_id':'Short Identifier'}

        def plan_details(self):
            details={}
            for line in self.read_lines():
                for k,v in self.__search_for.items():
                    if v in line:
                        item_details = line.split('=')[1].strip('\n')

                        if 'File' in line:
                            "If data is a file extension, return the full path"
                            details[k] = self.path.replace(self.path_extension, item_details)
                        else:
                            details[k] = item_details

            return details

        self.data = plan_details(self)

    def read_lines(self):
        with open(self.path, 'r') as f:
            return f.readlines()

class RasModel(object):
    '''
    HEC-RAS Model Object
    This object holds information for the files needed in a hec-ras model.

    If path starts with s3 then the code will run from s3 files and not local
    example: s3://pfra/dev/models/pluvial/DevMod1.prj
    '''
    def __init__(self, path, verbose=False):

        #Returns either a ZipFile object or string depending on the input path
        def obj_path(self, path):
            if path.endswith('.zip'):
                zipf = getZip(path)
                return zipf
            else:
                return path
        #Returns list of hdf files in a directory or zip
        def get_hdf_plan(self):

            if type(self.path) == str:
                return get_files(self.path, [self.prj.current_plan+'.hdf', self.prj.current_plan+'.tmp.hdf'])

            elif type(self.path) == zipfile.ZipFile:
                return searchZip(self.path, [self.prj.current_plan+'.hdf', self.prj.current_plan+'.tmp.hdf'])
        #Returns a file name by changing the extension from .p to the input extension
        def get_file(self, planName, extension):
             file = planName.replace('.p', '.'+extension)
             return file

        #Set model properties
        self.path = obj_path(self, path)
        self.verbose = verbose
        self.errors =[]
        self.project_name = os.path.basename(path).split('.')[0]
        self.prj = RasProject(self.path, self.project_name, self.verbose)
        self.hdf_plans = get_hdf_plan(self)
        self.b_file = get_file(self, self.prj.current_plan,'b')
        self.c_file = get_file(self, self.prj.current_plan,'c')
        self.x_file = get_file(self, self.prj.current_plan,'x')
        self.mandatory_files = {'b': self.b_file,
                                'c': self.c_file,
                                'x': self.x_file,
                               }

        #check for prj errors
        for e in self.prj.errors:
            self.errors.append(e)
            if verbose: print('prj.error: {}'.format(e.Error))
    #Returns either a list of errors or True. This method checks for correctness in all needed files for the RasModel.
    def prereqFileChecks(self):
        if len(self.hdf_plans) !=1 :
            e = ErrorClass("Expected to find one .hdf file, found  {}".format(len(self.hdf_plans)))
            if self.verbose: print('Prerequisite File error: {}'.format(e.Error))
            self.errors.append(e)

        if self.prj.current_plan != None :
            for f in self.mandatory_files.keys():
                if not check_file(self.mandatory_files[f], self.path):
                    print(self.mandatory_files[f], self.path)
                    e = ErrorClass("Mandatory .{} file not found".format(f))
                    if self.verbose: print('Prerequisite File error: {}'.format(e.Error))
                    self.errors.append(e)
        else:
            self.errors.append(ErrorClass("No Current Plan in prj file {}".format(self.path)))
            self.errors.append(e)

        if len(searchZip(self.path, ["."+self.b_file.split('.')[-1]])) > 1:
            self.errors.append(ErrorClass("Too many B files in the zip folder."))
        if len(searchZip(self.path, ["."+self.c_file.split('.')[-1]])) > 1:
            self.errors.append(ErrorClass("Too many C files in the zip folder."))
        if len(searchZip(self.path, ["."+self.x_file.split('.')[-1]])) > 1:
            self.errors.append(ErrorClass("Too many X files in the zip folder."))


        if len(self.prj.list_geom) > 1:
            self.errors.append(ErrorClass("Too many Geom File listed in the prj file"))

        if len(self.prj.list_unsteadies) > 1:
            self.errors.append(ErrorClass("Too many Unsteady File lines listed in the prj file "))

        if len(self.prj.list_plans) > 1:
            self.errors.append(ErrorClass("Too many Plan File lines listed in the prj file"))

        if len(self.errors) >0:
            if self.verbose: print('{} Prerequisite File errors'.format(len(self.errors)))
            return self.errors

        else:
            if self.verbose: print('All Prerequisite Files found, continuing checks')
            return True
    #This method sets the hdf property in the RasModel. This should be run after the prereqFileCheck.
    def unsteadySimFiles(self):
        errors =[]
        sim_files=[]
        if self.verbose: print(self.hdf_plans[0], self.path)
        try:
            self.hdf = UnsteadyHDFFile(self.hdf_plans[0], self.path, self.verbose)

            for e in self.hdf.errors:
                if self.verbose: print('Unsteady Sim File error {}'.format(e.Error))
                self.errors.append(e)

            for k, v in self.mandatory_files.items():
                if self.verbose: print(k, v)
                sim_files.append(ComputeFile(v, self.path))
                #sim_files.append(v)
            self.sim_files = sim_files
            return sim_files

        except:
            print("Failure at model.unsteadySimFiles()")
            return False
    #Method to overwrite the current RasModel with any changes to the hdf property
    def save(self):
        self.hdf.saveHDF(self.hdf.path)

    #Method to save a new copy of the RasModel with any changes to the hdf property
    def saveAs(self, newPath):
        #TODO pass in either a path or generic file name and change
        #extensions as needed
        self.hdf.saveHDF(newPath)
    #Method to zip the RasModel files into a new zip folder.
    def zipModel(self, runName, hdfFile):
        print('creating archive')
        #create a file with updating data in b & x compute file to include in the zip
        bpath = self.mandatory_files["b"]
        bcompute =[x for x in self.sim_files if x.path == bpath]
        bfilename = '{}{}'.format(runName.replace('_in',''), pl.Path(bpath).suffix)
        with open(bfilename, 'w') as f:
            for item in bcompute[0].data:
                f.write("%s\n" % item)

        xpath = self.mandatory_files["x"]
        xcompute =[x for x in self.sim_files if x.path == xpath]
        xfilename = '{}{}'.format(runName.replace('_in',''), pl.Path(xpath).suffix)
        with open(xfilename, 'w') as f:
            for item in xcompute[0].data:
                f.write("%s\n" % item)

        with zipfile.ZipFile('{}.zip'.format(runName) , mode='w') as zf:
            cFile = self.mandatory_files["c"];
            zf.writestr('{}{}'.format(runName.replace('_in',''),    pl.Path(cFile).suffix), self.path.read(self.path.NameToInfo[cFile]))

            zf.write(bfilename, pl.Path(bfilename).name)
            zf.write(bfilename, pl.Path(xfilename).name)
            zf.write(hdfFile, pl.Path(hdfFile).name)
        zf.close()

        os.remove(bfilename)
        os.remove(xfilename)


class UnsteadyHDFFile(object):
    '''
    HEC-RAS Unsteady HDF Plan File Object
    This object holds information for items contained in the HDF plan file.
    '''
    def __init__(self, path, zip, verbose=False):

        #Method to fix decrepencies between slashes
        def prep_paths(self):
            if type(path) == str:
                class_path = path.replace('\\','/')
            else:
                class_path = path.filename
            return class_path
        #Method to set the local path of the hdf file. This is needed when the file has been download from s3.
        def local_path(self):
            if type(path) == str:
                if path.replace('\\','/').startswith('s3://'):
                    localPath = os.path.join(os.getcwd(), os.path.basename(path))
                else:
                    localPath = path;

            elif type(path) == zipfile.ZipInfo:
                 localPath = os.path.join(os.getcwd(), path.filename)
            return localPath

        self.path = prep_paths(self)
        self.localPath  = local_path(self)
        self.errors = []
        self.verbose=verbose
        self.key_names = {}
        self.hydrograph_parent_path = 'Event Conditions/Unsteady/Boundary Conditions'
        #Sets the HDF file property which is of type h5py.File
        try:
            self.hdf = openHDFFile(path, zip)
        except:
            self.errors.append(ErrorClass(':Failure opening hdf file'))

        #Example of how to descend through all attributes and objects in an h5py.File, used as reference.
        #def descend_obj(obj,sep='\t'):
        #    """
        #    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes, using this as reference
        #    """
        #    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        #        for key in obj.keys():
        #            print(sep,'-',key,':',obj[key])
        #            descend_obj(obj[key],sep=sep+'\t')
        #    elif type(obj)==h5py._hl.dataset.Dataset:
        #        for key in obj.attrs.keys():
        #             print(sep+'\t','-',key,':',obj.attrs[key])
        #descend_obj(self.hdf)

        def get_hydrograph_keys(self):
            hkeys = [x for x in self.hdf[self.hydrograph_parent_path].keys() if str.upper(x).find("HYDROGRAPH") >= 0]
            return hkeys

        def get_domain_path(self):
            domain_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/'
            return domain_path

        def get_domain_names(hf):
            domain_paths = get_domain_path(self)
            domains = list(hf[domain_paths].keys())
            return domains
        def populateHydrographs(self):
            try:
                graphs={}
                # key_names = {}
                h = get_hydrograph_keys(self)
                for v in h:
                    hyd = self.hdf['Event Conditions/Unsteady/Boundary Conditions'][v]
                    for f in hyd:
                        # key_names[v] = f
                        graph = hyd[f]
                        graphs[f] = graph

                return graphs
            except:
                e = ErrorClass('Failure opening Boundary Conditions for {}'.format(self.path))
                if self.verbose: print('Populate Hydrographs error: {}'.format(e.Error))
                self.errors.append(e)

        def getCellCount(self):
            cell_count = -9999
            try:
                cell_count_key = [x for x in self.hdf['Geometry/2D Flow Areas'].keys() if str.upper(x).find("ATTRIBUTES") >= 0]
                cell_count = self.hdf['Geometry/2D Flow Areas'][cell_count_key[0]]['Cell Count'][0]
            except:
                e = ErrorClass('Failure setting cell count for {}'.format(self.path))
                if self.verbose: print('Populate Cell Count error: {}'.format(e.Error))
                self.errors.append(e)


            if len(cell_count_key) == 0:
                e = ErrorClass('No Attributes exist for {}'.format(self.path))
                if self.verbose: print('Populate Cell Count error: {}'.format(e.Error))
                self.errors.append(e)
            return cell_count


        try:
           self.Domains = get_domain_names(self.hdf)
           self.Hydrographs = populateHydrographs(self)
           self.cell_count = getCellCount(self)
           self.hkeys = get_hydrograph_keys(self)
        except:
            e = ErrorClass(':Error reading domains and domain errors {}'.format(self.path))
            if self.verbose: print('Populate Hydrographs error: {}'.format(e.Error))
            self.errors.append(e)



    def setHydrographData(self, key, data):
        try:
            modelHydrographs = self.Hydrographs.copy()
            hkeys = self.hkeys
            current_key = [key_x for key_x in hkeys if key in self.hdf[self.hydrograph_parent_path][key_x].keys()][0]
            full_group_path = self.hydrograph_parent_path + '/' + current_key
            # print(full_group_path)
            del self.hdf[full_group_path]
            del self.Hydrographs[key]
            hyd = self.hdf[self.hydrograph_parent_path].create_group(current_key)
            hyd.create_dataset(key, (len(data), 2), maxshape=(None, 2), dtype='float32')

            for attribute in modelHydrographs[key].attrs.keys():

                hyd[key].attrs.create(attribute, modelHydrographs[key].attrs[attribute], None)

            self.Hydrographs[key] = hyd[key]

            try:
                # self.Hydrographs[key] = self.hdf.create_dataset('test_dataset', (len(data),2), maxshape=(None,2))
                self.Hydrographs[key].resize((len(data), 2))
                self.Hydrographs[key][:] = data
            except:
                e = ErrorClass('Failure opening Boundary Conditions for {}'.format(self.path))
                if self.verbose: print('Populate Hydrographs error: {}'.format(e.Error))
                self.errors.append(e)
        except:
            e = ErrorClass('Failure finding Hydrograph for {}'.format(self.path))
            if self.verbose: print('Populate Hydrographs error: {}, {} not found in model'.format(e.Error, key))
            self.errors.append(e)
    # Functions from qaqc.py
    def comp_msgs(self):
        binary_data = np.array(self.hdf['Results/Summary/Compute Messages (text)'])
        messages = [data.decode('UTF-8').strip() for data in binary_data]
        messages=messages[0].split('\r')
        msg = [m.strip('\n') for m in messages if (len(m) > 0) and (m.strip('\n')[0:4] != 'File')]
        return msg

    def iterations_2d(self, domain:str):
        iterations = np.array(self.hdf[f'/Results/Unsteady/Output/Output Blocks/Computation Block/2D Flow Areas/{domain}/2D Iterations'])
        iteration_errors = np.array(self.hdf[f'/Results/Unsteady/Output/Output Blocks/Computation Block/2D Flow Areas/{domain}/2D Iteration Error'])
        return iterations, iteration_errors

    def get_2d_errors(self):
        domain_errors = {}
        for d in self.Domains:
            iterations, iteration_errors  = self.iterations_2d(d)
            domain_errors[d] = iteration_errors.sum()

        return domain_errors
    def boundary_conditions_domain(self, domain:str):
        figs=[]
        df = pd.DataFrame()
        bcdata={}
        bcs = np.array(self.hdf[f'/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{domain}/Boundary Conditions/'])
        for bc in bcs:
            bcdata[bc] = np.array(self.hdf[f'/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{domain}/Boundary Conditions/{bc}'])
            if 'Flow' in bc:
                df[bc] = bcdata[bc].sum(axis=1)
                fig, ax = plt.subplots(figsize=(20,2))
                ax.plot(df[bc], color='blue', label=bc);ax.legend(),ax.grid()
            elif 'Stage' in bc:
                df[bc] = bcdata[bc].mean(axis=1)
                fig, ax = plt.subplots(figsize=(20,2))
                ax.plot(df[bc],color='black', label=bc);ax.legend(),ax.grid()

            figs.append(fig)

        bcdata = df.to_dict()
        return figs, bcdata
    def get_mannings_data(self):
        binary_data = np.array(self.hdf[r"/Geometry/Land Cover (Manning's n)/Calibration Table"])
        return dict((x,y) for x, y in binary_data)

    def maxv(self, domain):
        binary_data = np.array(self.hdf[f'/Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/{domain}/Maximum Face Velocity'])
        return binary_data

    def minv(self, domain):
        binary_data = np.array(self.hdf[f'/Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/{domain}/Minimum Face Velocity'])
        return binary_data

    def get_velocities(self):
        domain_velocities = {}

        for d in self.Domains:
            v_stats = {}
            print(d)

            maxvel = self.maxv(d)
            minvel = self.minv(d)

            v_stats['vgt20'] = np.sum(abs(maxvel[0]) > 20)
            v_stats['vgt50'] = np.sum(abs(maxvel[0]) > 50)
            v_stats['vgt100'] = np.sum(abs(maxvel[0]) > 100)
            v_stats['maxv'] = abs(maxvel[0]).max()

            domain_velocities[d] = v_stats

            fig, ax = plt.subplots(figsize=(20,4))
            ax.plot(maxvel[0], color='red');ax.set_title(d);ax.set_ylabel('Velocity (ft/s)');ax.set_xlabel('Domain Cell Faces')
            ax.plot(minvel[0], color='black');ax.set_title(d);ax.set_ylabel('Velocity (ft/s)');ax.set_xlabel('Domain Cell Faces')
            ax.grid()

        return domain_velocities

    def get_structure_names(self, print_path=False):
        structs = {}
        structure_paths = ['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/SA 2D Area Conn',
                           'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Lateral Structures']
        for path in structure_paths:
                try:
                    structures = np.array(self.hdf[path])
                    for structure_found in structures:
                        if print_path: print(f'Structure: {path}/{structure_found}')
                        structs[structure_found] = f'{path}/{structure_found}'
                except:
                    continue
        return structs

    def read_plan(self):
        _attrs        = '/Geometry/Structures/Attributes'
        _cline_info   = '/Geometry/Structures/Centerline Info'
        _cline_parts  = '/Geometry/Structures/Centerline Parts'
        _cline_points = '/Geometry/Structures/Centerline Points'
        _profiles     = '/Geometry/Structures/Profiles'

        attrs         = pd.DataFrame(np.array(self.hdf[f'{_attrs}']), dtype=str)
        cline_info    = pd.DataFrame(np.array(self.hdf[f'{_cline_info}']), dtype=str)
        cline_info.rename(columns={0:'Start', 1:'nrows'}, inplace=True)
        cline_info    = cline_info[['Start', 'nrows']].apply(pd.to_numeric)

        attrs.index.name = 'StructID'
        cline_info.index.name='StructID'

        cline_parts  = pd.DataFrame(np.array(self.hdf[f'{_cline_parts}']), dtype=str)
        cline_points  = pd.DataFrame(np.array(self.hdf[f'{_cline_points}']), dtype=str)
        profiles      = pd.DataFrame(np.array(self.hdf[f'{_profiles}']), dtype=str)

        return attrs, cline_info, cline_points, profiles, cline_parts

    def updated_structure_data(self, attrs, cline_info, cline_points, profiles, cline_parts, structures):
        for i, idx in enumerate(attrs.index):
            stype = attrs.loc[idx,'Type']
            sriver = attrs.loc[idx,'River']
            sreach = attrs.loc[idx,'Reach']
            srs    =attrs.loc[idx,'RS']
            sconnect = attrs.loc[idx,'Connection']
            sus_bound = attrs.loc[idx,'US SA/2D']
            sds_bound = attrs.loc[idx,'DS SA/2D']

            if stype == 'Lateral':
                structname =  sriver+' '+sreach +' '+srs
            elif stype == 'Connection' and sus_bound==sds_bound:
                structname =  sus_bound+ ' '+sconnect
            elif stype == 'Connection' and sus_bound!=sds_bound:
                structname =  sconnect

            if structname in structures:
                geox = cline_info.loc[idx, 'Start']
                geoy = cline_info.loc[idx, 'nrows']+ geox
                profile = profiles[profiles['SID']==str(idx)].copy()

                flow, wse_hw, wse_tw, hw_cells, tw_cells , hw_tw_stns = get_structure_data(rasplan, structname, structures[structname])

                try:
                    # Break Function here
                    fig, ax = plt.subplots(figsize=(28,4))

                    x, y = profile['Station'].apply(pd.to_numeric), profile['Elevation'].apply(pd.to_numeric)
                    struct_geom = pd.DataFrame(y.reset_index(drop=True).values, index=x.reset_index(drop=True).values)
                    ax.plot(struct_geom.index,struct_geom[0].values, color='grey', label='Levee Crest');ax.set_title(structname)

                    # Plot Head Water
                    cell_wse_hw = pd.DataFrame(np.amax(wse_hw, axis=0),hw_tw_stns)
                    ax.plot(cell_wse_hw.index,cell_wse_hw[0].values, label = 'Max HW', color = 'blue');ax.set_title(structname)

                    # Plot Tail Water
                    #cell_wse_tw = pd.DataFrame(np.amax(wse_tw, axis=0),hw_tw_stns)
                    #ax.plot(cell_wse_tw.index,cell_wse_tw[0].values, label = 'Max TW', color = 'black');ax.set_title(structname)

                    # Fill Plots
                    ylim = np.min([wse_tw.min(),wse_hw.min()])
                    ax.fill_between(x, ylim,y , color='grey', alpha=0.5)
                    ax.fill_between(hw_tw_stns, ylim, np.amax(wse_hw, axis=0) , color='blue', alpha=0.2)
                    #ax.fill_between(hw_tw_stns, ylim, np.amax(wse_tw, axis=0) , color='black', alpha=0.1)
                    ax.legend()
                except:
                    print('ERROR', structname, 'No Connection')#, type(hw_tw_stns), hw_tw_stns.shape)
                    continue

#Method to save or overwrite the hdf file. Changes to the Hydrograph objects will be saved. Results will not be carried over.
    def saveHDF(self, newPath):
            print("Saving file: " + newPath)
            hdf = h5py.File(newPath, 'w')
            org = self.hdf
            #copy original information to new file
            for fg in org.keys():
                if fg not in["Results"]:
                    org.copy(fg, hdf)
            assert hdf != None, 'Error saving  HDF file {}'.format(newPath)

            #save work
            if hdf != None:
                hdf.close()
            return newPath

class BoundaryCondition(object):
      def __init__(self, domain, figs, bcdata):

            self.Domain = domain
            self.Figures = figs
            self.Data = bcdata

class ComputeFile(object):
    '''
       HEC-RAS Compute File
       Add other items as needed to search_for dict
    '''
    def __init__(self, path, zipFile):
        try:
            self.errors=[]
            self.path = path
            self.path_extension = path.split('.')[-1]

            #c is a binary file so skip this for now
            if str.startswith(self.path_extension, 'c') == False:
                self.data = self.read_lines(zipFile)
        except:
            e = ErrorClass('Error setting compute file {}'.format(self.path))
            self.errors.append(e)

    def update_data_index(self, index, newVal):
        self.data[index] = newVal

    def get_line_index(self, value):
        return self.data.index(value)

    def get_line_with_prefix(self, value):
        index = [i for i, s in enumerate(self.data) if s.startswith(value)]
        return index[0]

    def read_lines(self, zipFile):
        files = searchZip(zipFile, ["."+self.path_extension])
        return zipFile.read(files[0]).decode('utf-8').splitlines()


class ErrorClass(object):
    def __init__(self, error):
        self.Error = error

#--------------------------------FUNCTIONS-------------------------------------#

#----Getters----#

#Returns a list of files with the specified extension in a directory, including s3.
def get_files(path, ext):
    try:
        if path.replace('\\','/').startswith('s3://'):
            bucket, key = splitS3Path(path)
            files = []
            bpath, bfilename = os.path.split(path)
            for s3_object in s3.Bucket(bucket).objects.filter(Prefix= os.path.dirname(key)):
                path, filename = os.path.split(s3_object.key)
                files.append(os.path.join(bpath, filename))
            foundFiles = [x for x in files for y in ext if x.replace('\\','/')== y]
            return foundFiles
        else:
            directory= os.path.dirname(path)
            foundFiles =[]
            for e in ext:
                foundFiles.append(os.path.join(directory, '*.{}'.format(e)))
            return foundFiles
    except:
         return (ErrorClass(':Failure getting files in {}'.format(path)))
#Returns zipfile data from s3
def getZip(path: str):
    bucket, key = splitS3Path(path)

    obj = s3.Object(
        bucket_name=bucket,
        key=key
    )
    buffer = io.BytesIO(obj.get()["Body"].read())
    z = zipfile.ZipFile(buffer)

    return z

#Returns a dictionary of path parts that are used throughout scaling tests
def getPaths(path):
    dir_path = os.path.dirname(path)
    fileName = os.path.basename(path).split('.')[0]
    fileParts = fileName.split('_')
    projectName = fileParts[0]
    subType = fileParts[1]
    h0 = fileParts[2]
    jsonPath = os.path.join(dir_path, "{}_{}_{}.zip".format(projectName, subType, "forcing")).replace('\\','/')
    errorFilePath = os.path.join(dir_path, "Error_{}_{}.txt".format(projectName, subType)).replace('\\','/')
    path_data = {'dir_path':dir_path,
                'fileName': fileName,
                'fileParts':fileParts,
                'projectName':projectName,
                'subType':subType,
                'h0':h0,
                'jsonPath':jsonPath,
                'errorFilePath':errorFilePath}
    return path_data
#Returns a list of string names of events in a json file.
def getAllEventsFromJSON(data):
    events=[]
    for key, value in data.items():
            for key1, value1 in data[key]['BCName'].items():
                for key2, value2 in data[key]['BCName'][key1].items():
                    if key2 not in events:
                        events.append(key2)
    return events
def getAllProductionEventsFromJSON(jsonData, scaleEvents):
    events=[]
    for d in jsonData:
        data = json.loads(d)
        for key, value in data.items():
                for key1, value1 in data[key]['BCName'].items():
                    for key2, value2 in data[key]['BCName'][key1].items():
                        if key2 not in scaleEvents:
                            if key2 not in events:
                                events.append(key2)
    return events
#----Checkers----#
#Function to check for a file's existance. Returns boolean.
def check_file(path, fullPath='', verbose=False):
    try:
        if path.replace('\\','/').startswith('s3://') :
                bucket, key = splitS3Path(path)
                try:
                    s3.Object(bucket, key).get()
                    return True
                except:
                    return False
        elif type(fullPath) == zipfile.ZipFile:
                return checkZip(fullPath, path)
        else:
            return os.path.isfile(path);
    except:
         return False
#Function to check a zipfile.Zipfile contents for a specific file. Returns boolean.
def checkZip(z: zipfile.ZipFile, name: str, verbose=False):
    print('Looking for {}'.format(name.upper()))
    file = [x for x in z.infolist() if str.upper(x.filename) == name.upper()]
    if verbose:
        print('Zip Contents:\n')
        for x in z.infolist():
             print(str.upper(x.filename))

    if(len(file) == 1):
        return True
    else:
        return False
#Function to check the input json for scaling tests for correctness. This function will also check the hydrograph objects in the input RASModel for matching Boundary Condition names in the json file.
#Returns a list of errors
def checkJSON(path:str, model:RasModel, verbose=False):
    errors=[]
    try:
        bucket, key = splitS3Path(path)
        s3_obj = s3.Object(bucket, key)
        info = s3_obj.get()["Body"].read().decode('utf-8')
        data = json.loads(info)
        bcNames = []
        for key, value in data.items():
            for key1, value1 in data[key]['BCName'].items():
                    if key1 not in bcNames:
                        bcNames.append(key1)

        for bc in bcNames:
            if verbose: print('Checking bc: {}'.format(bc))
            try:
                #TODO: will SA always be there? might need to check 2D Flow Area attribute instead
                #print(model.hdf.Hydrographs.keys())
                ph = [x for x in model.hdf.Hydrographs.keys() if str.upper(x) == str.upper('SA: '+ bc)]
                #print(ph)
                if len(ph) ==0:
                    e = ErrorClass('A hydrograph for {} does not exist'.format(bc))
                    if verbose: print('checkJSON error: {}'.format(e.Error))
                    errors.append(e)

            except:
                e = ErrorClass('Model object is not complete, updates will not be written')
                if verbose: print('checkJSON error: {}'.format(e.Error))
                errors.append(e)
        return errors

    except:
        e = ErrorClass('Error parsing JSON {}'.format(path))
        if verbose: print('checkJSON error: {}'.format(e.Error))
        errors.append(e)
        return errors
#Function to save error list to s3 if the list has contents.
def checkErrors(errors, errorFilePath):
    bool_error = False
    if len(errors) > 0:
        bool_error = True
        try:
            print("Scaling will not continue due to the following {} errors:".format(len(errors)))
            errorList = [e.Error for e in errors]
            for e in errorList:
                print(e)
            print("Errors are being saved to: {}".format(errorFilePath))
            nxtlne = os.linesep
            errorFile = nxtlne.join(errorList)
            f = io.BytesIO(errorFile.encode())
            saveBinaryToS3(f, errorFilePath)

        except Exception as e:
            print("FAILURE SAVING ERRORS: {}".format(e.args[0]))
        return bool_error
    else:
        print('No errors found')
        return bool_error

def checkS3ForFile(path):
    bucket, key = splitS3Path(path.replace('\\','/'))
    s3_obj = s3.Object(bucket, key)
    spath, filename = os.path.split(s3_obj.key)
    my_bucket = s3.Bucket(bucket)
    objs = list(my_bucket.objects.filter(Prefix=key))
    if len(objs) > 0 and objs[0].key == key:
        return True
    else:
        return False


def checkIfS3EventsExist(events, projectName, subType):

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


#----Searcher-Finders----#

def find_ras(directory:str):
    '''Search Directory for Ras Model, expect to only find one.
       For use in PFRA scaling module.
    '''
    prj_files = glob(os.path.join(directory, '*.prj'))
    project_files = []
    for prj in prj_files:
        with open(prj, 'r') as f:
            lines = f.readlines()
            if 'Proj Title' in lines[0]:
                project_file = prj
                project_files.append(project_file)

    return project_files
#Function to search a zip file for a file with a specific extension. Returns a list.
def searchZip(z: zipfile.ZipFile, ext):

    files = [x for x in z.infolist() for y in ext if str.endswith(x.filename, y)]
    #for f in file:
    #    foo2 = z.open(f)
    return files

#----Openers----#
#Function to open an HDF file with h5py from s3 or locally.
def openHDFFile(path, zip):
    if type(path) == str:
        if path.replace('\\','/').startswith('s3://'):
            bucket, key = splitS3Path(path.replace('\\','/'))
            s3_obj = s3.Object(bucket, key)
            spath, filename = os.path.split(s3_obj.key)
            my_bucket = s3.Bucket(bucket)
            print('Downloading: {}'.format(filename))
            my_bucket.download_file(s3_obj.key, filename)
            return h5py.File(filename, 'r+')
        else:
            return h5py.File(path, 'r+')
    elif type(path) == zipfile.ZipInfo:
        zip.extract(path)
        return h5py.File(path.filename, 'r+')


#----Updaters----#
def saveModelFilesToS3(model, projectName, subType, event, interval, localOutputPath, runs):
    cell_count_value = model.hdf.cell_count
    baseFile = (os.path.splitext(os.path.basename(model.hdf.localPath))[0]).split('.')[0]
    newFileName =  '{}_{}_{}_{}.{}.tmp.hdf'.format(projectName, subType, interval, event, model.prj.current_plan_ext)
    zipName = '{}_{}_{}_{}_in'.format(projectName, subType, interval, event)
    localHDFName = localOutputPath+"\\"+newFileName
    model.saveAs(localHDFName)
    model.zipModel(zipName, localHDFName)
    s3Path = 's3://pfra/{}/{}/{}/{}/{}.zip'.format(projectName, subType, interval, event, zipName)
    print("Saving file to s3 {}".format(s3Path))
    saveFileToS3(zipName+'.zip', s3Path)
    os.remove(localHDFName);
    os.remove(zipName+'.zip');
    runs.write(s3Path+','+cell_count_value.astype(str)+'\n')

def updateComputeFile(model: RasModel, projectName:str, planName:str, dayDuration:int):
    bpath = model.mandatory_files["b"]
    bcompute =[x for x in model.sim_files if x.path == bpath][0]

    xpath = model.mandatory_files["x"]
    xcompute =[x for x in model.sim_files if x.path == xpath][0]

    index = bcompute.get_line_index("Project Title, Plan Title and Plan ShortID")
    bcompute.update_data_index(index+1, projectName)
    bcompute.update_data_index(index+2, planName)
    bcompute.update_data_index(index+3, planName)

    xcompute.update_data_index(2, planName)

    index = bcompute.get_line_with_prefix("  Start Date/Time")
    bcompute.update_data_index(index, "  Start Date/Time       = {} 2400".format("01May2000"));
    day = 1+ dayDuration

    index = bcompute.get_line_with_prefix("  End Date/Time")
    newDate ="{}May2000".format('%02d' % day)
    bcompute.update_data_index(index, "  End Date/Time         = {} 2400".format(newDate));

#Function to update hydrograph data for a specified event, if any of the hydrographs are changed then it is saved up to s3
def updateModelHydrographsForInterval(interval, data, events, model, projectName, subType,localOutputPath, runs):
    # grab json data for times and boundary conditions
    dates = data[interval]['time_idx']
    hgraphs = data[interval]['BCName']
    duration = data[interval]['run_duration_days']

    #loop through events
    for event in events:
        updated = False #reset flag
        #Loop through each boundary condition found in the json file
        for bcname in hgraphs:
            #Check if the current event is in the json data for the specified interval and boundary condition name
            if event in data[interval]['BCName'][bcname]:
                eventData = data[interval]['BCName'][bcname][event]
                #Find the hydrograph that matches the current boundary condition name, this should only be one hydrograph
                hydrographs = model.hdf.Hydrographs
                bc_name_updated = bcname

                if bcname[0] == 'D':
                    bc_name_updated = 'SA: ' + bcname
                ph = [x for x in hydrographs.keys() if str.upper(x) == str.upper(bc_name_updated)]
                if len(ph) > 0:
                    timestamp_adj = np.divide(dates, 24)
                    graph_data = list(map(list, zip(timestamp_adj, eventData)))

                    model.hdf.setHydrographData(bc_name_updated, graph_data)
                    updated = True

        if updated: #If a hydrograph was updated then save the model updates to s3
            projName = '{}_{}_{}_{}'.format(projectName, subType, interval, event)
            updateComputeFile(model, projName, event, duration)
            saveModelFilesToS3(model, projectName, subType, event, interval,localOutputPath, runs)

#Function to update a RASModel object using a json file from s3. Returns a csv file.
def updateModelFromJSON(jsonFilePath:str, events:list, model:RasModel, localOutputPath:str,
                        projectName:str, subType:str, custom_events:bool=False, prodRun:bool = False):
    '''
    Inputs:
        jsonFilePath: e.g. 's3://pfra/atkinstest/BaseModels/atkinstest_P01_H00.json'
        events: List of specified events to simulate, leave blank to run scale tests or all production runs
        model: RasModel Object
        localOutputPath: local directory to temporarily read/write hdf & other ras data files
        projectName: name of project study area (e.g. Skokie)
        subType: Fluvial or Pluvial
        custom_events: Specify if custom events should be run.  If set to True must pass list to events (e.g.  events =  ['E0034', 'E00125'])
    Output:
        runsFile: csv written to s3 which lists the  s3 paths of models ready to be run by a scaling group
    '''
    #bucket, key = splitS3Path(jsonFilePath)
    #s3_obj = s3.Object(bucket, key)
    #info = s3_obj.get()["Body"].read().decode('utf-8')

    jsonZip = getZip(jsonFilePath)
    jsonFiles = searchZip(jsonZip, [".json"])
    jsonData = []
    for j in jsonFiles:
        jsonData.append(jsonZip.read(j))


    runsFile = 'runs_{}.csv'.format(time.ctime().replace(':','-').replace('  ',' ').replace(' ','_'))

    # Get top level key for parsing json ('interval')
    if subType.startswith('P'):
        intervals = ['H06', 'H12', 'H24', 'H96']

    elif subType.startswith('F'):
        intervals = ['Fluvial']

    # If scale test, run scale events
    if (len(events)==0) and not prodRun:
        assert(len(events) > 1), "scale test error on updateModelFromJSON"

    # If prod run and no events specified, run all except scale events
    elif custom_events:
        assert(len(events) > 1), "Must specify events"

    # If prod run and no events specified, run all except scale events
    elif prodRun and not custom_events:
        # Run all evelts
        events = getAllProductionEventsFromJSON(jsonData, scaleTestEvents(jsonFilePath))

    with open(runsFile, 'w') as runs:
        #loop through all json files
        for d in jsonData:
            data = json.loads(d)
            #loop through all intervals and update hydrographs if possible
            for interval in intervals:
                if interval in data:
                    if interval=='Fluvial':
                        # Fluvial doesnt distinguish by Hour, but by Breach ID
                        # Use model_identifier to pull this from json path string temporarily
                        model_identifier = jsonFilePath.split('_')[-1].split('.')[0]
                        updateModelHydrographsForInterval(model_identifier, data, events, model, projectName, subType, localOutputPath, runs)
                    else:
                        updateModelHydrographsForInterval(interval, data, events, model, projectName, subType, localOutputPath, runs)


    model.hdf.hdf.close()
    os.remove(model.hdf.localPath)
    return runsFile

#----Readers----#

def read_lines(path):
        if path.replace('\\','/').startswith('s3://'):
            bucket, key = splitS3Path(path)
            s3_obj = s3.Object(bucket, key)
            info = s3_obj.get()["Body"].read().decode('utf-8')
            return info.splitlines()
        else:
            with open(path, 'r') as f:
                return f.readlines()

#----Splitters----#

def splitS3Path(path):
    prefix, filename = os.path.split(path)
    parts = path.replace('\\','/').split('/')
    bucket = parts[2]
    key = str.replace(path,'s3://{}'.format(bucket), '')
    return bucket, key[1:]

#----Downloaders----#
def downLoadFiles(s3Path : str, newPath:str):

    os.system("aws s3 cp "+s3Path+" "+newPath+" --recursive")

#----Savers----#

def saveFileToS3(file, folder):
    bucket, key = splitS3Path(folder.replace('\\','/'))
    s3.Bucket(bucket).upload_file(file, key)

def saveBinaryToS3(file, folder):
    bucket, key = splitS3Path(folder.replace('\\','/'))
    s3.Bucket(bucket).upload_fileobj(file, key)

#----Testers----#

def scaleTestEvents(path):
    scale_events = {'Pluvial': ["E0001", "E0201", "E1001", "E1201",
                               "E2001", "E2201", "E3001", "E3201"],
                       'Fluvial': ["E0001", "E0025", "E0075", "E0100"]}

    if 'P' in pl.Path(path).name.split('_')[1]:
        return scale_events['Pluvial']
    elif 'F' in pl.Path(path).name.split('_')[1]:
        return scale_events['Fluvial']
    else:
        print('Error in Path, unable to detect run type')
        return None

# Add check to see if inputs are on s3 for production runs
# e.g. if scale test was successful, don't redo those during production
