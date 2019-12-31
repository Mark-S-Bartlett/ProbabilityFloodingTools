import h5py
import numpy as np
import pathlib as pl
import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import OrderedDict
from pathlib import PurePosixPath as posix
import matplotlib.pyplot as plt

class rasHDF(object):
    '''
    HEC-RAS HDF Plan File Object to compute flow data at breaklines.
    Some functionality may be useful for other ras objects.
    '''
    def __init__(self, path, domain, verbose=False):

        # Specify Domain to instantiate Object
        self.path = path
        self.domain = domain

        def get_top_level_data(self, data):
            '''Inventory top level of ras planfile'''
            with h5py.File(self.path,'r') as hf:
                members = list(hf[data].keys())
            return ['{}/{}'.format(data,m) for m in members]

        def get_breakline_names(self):
            '''List names of breaklines in 2D flow areas'''
            member = 'Geometry/2D Flow Area Break Lines/Attributes'
            with h5py.File(self.path,'r') as hf:
                table = np.array(hf['{}'.format(member)])
            return [t[0].decode() for t in table]

        def get_breakline_coords(self, bLines):
            '''Return a dictionary of coordinates for each breakline'''
            member = 'Geometry/2D Flow Area Break Lines'
            with h5py.File(self.path,'r') as hf:
                info = np.array(hf['{}/{}'.format(member, 'Polyline Info')])
                points = np.array(hf['{}/{}'.format(member, 'Polyline Points')])
                bLinePointCoords = {}
                for idx, b in enumerate(bLines):
                    start = info[idx,0]
                    stop = start + info[idx,1]
                    bLinePointCoords[b] = points[start:stop]
            return  bLinePointCoords
        
        # Get all Top Level Data, Breakline & Domains at init
        # Add Warning/KeyError if data not found?
        self._event  = get_top_level_data(self, 'Event Conditions')
        self._geometry  = get_top_level_data(self, 'Geometry')
        self._plan  = get_top_level_data(self, 'Plan Data')
        self._results  = get_top_level_data(self, 'Results')

        self._domain_path = 'Geometry/2D Flow Areas/{}'.format(self.domain)
        self._blines = get_breakline_names(self)
        self._bline_coords = get_breakline_coords(self, self.blines)


    '''
    Getter functions for dev/debug.
    '''
    @property
    def geometry_tables(self):
        return self._geometry_tables

    @property
    def event(self):
        return self._event

    @property
    def geometry(self):
        return self._geometry

    @property
    def plan(self):
        return self._plan

    @property
    def results(self):
        return self._results

    @property
    def domain_path(self):
        return self._domain_path

    @property
    def blines(self):
        return self._blines

    @property
    def blineCoords(self):
        return self._bline_coords

    def list_breakline_tables(self, member = 'Geometry/2D Flow Area Break Lines'):
        with h5py.File(self.path,'r') as hf:
            tables = np.array(hf[member])
        return  ['{}'.format(t).replace(' ','_') for t in tables]

    def list_model_domains(self, data):
        other_data = ['Attributes', 'Cell Info','Cell Points','Polygon Info',
                      'Polygon Parts','Polygon Points']
        with h5py.File(self.path,'r') as hf:
            members = list(hf[data].keys())
        return ['{}/{}'.format(data,m) for m in members if m not in other_data]

    def list_domain_tables(self, domain):
        with h5py.File(self.path,'r') as hf:
            tables = np.array(hf[domain])
        return  ['{}'.format(t).replace(' ','_') for t in tables]

    def domain_table(self, domain, table, table_type='data'):
        '''Return dictionary with self-describing data for 2D data
           default table_type returns data only. 
        '''
        with h5py.File(self.path,'r') as hf:
            name = '{}/{}'.format(domain, table.replace('_',' '))
            data = np.array(hf[name].value)
            rowData = np.array(hf[name].attrs['Row'])
            colData = np.array(hf[name].attrs['Column'])
        if table_type=='data':
            return data
        else:
            return  {'table':table, 'data':data, 'rowData':rowData, 'colData':colData}


    def fpoints_at_bline(self, fPointCoords):
        '''Build a map of faceline points given breakline points'''
        fPointIDs_atBLine = {}
        for bline, coords in self._bline_coords.items():
            fPointIDs_atBLine[bline] = []
            for point in coords:
#                 print("point: ", point, " fpoint: ", fPointCoords)
                idx = np.where(fPointCoords == point)
                try:
                    assert idx[0][0] == idx[0][1], 'Points not in the same array'
                    fpointIdx = idx[0][0]
                    fPointIDs_atBLine[bline].append(fpointIdx)
#                     print('\tFace Point ID = ', fpointIdx)
                except:
                    continue
        return fPointIDs_atBLine
    
    def bline_filter(self, fpoints_at_bline):
        '''Filter Breakline coordinates dataset to isolate those points that coincide
           with faceline points. Return a list of appropriately placed breaklines where all points
           coincide with faceline points, and a list of breaklines where flow cannot be calcuated.
        '''
        good_bLines, bad_bLines = [],[]
        for k,v in self._bline_coords.items():
            print(v.shape[0], len(fpoints_at_bline[k]))
            if (v.shape[0] == len(fpoints_at_bline[k])) and (v.shape[0] > 1) :
                good_bLines.append(k)
            else:
                bad_bLines.append(k)
        return good_bLines, bad_bLines

    def unsteady_results(self, domainName, table):
        '''Read in data from results tables'''
        with h5py.File(self.path,'r') as hf:
            name = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{}/{}'.format(domainName, table)
            return  np.array(hf[name].value)

    '''
    Getter functions for required tables
    '''
    @property
    def Faces_FacePoint_Indexes(self):
        return self.domain_table(self._domain_path, 'Faces_FacePoint_Indexes')

    @property
    def Cells_Minimum_Elevation(self):
        return self.domain_table(self._domain_path,'Cells_Minimum_Elevation')

    @property
    def FacePoints_Face_and_Orientation_Info(self):
        return self.domain_table(self._domain_path, 'FacePoints_Face_and_Orientation_Info')

    @property
    def FacePoints_Face_and_Orientation_Values(self):
        return self.domain_table(self._domain_path, 'FacePoints_Face_and_Orientation_Values')

    @property
    def Faces_Cell_Indexes(self):
        return self.domain_table(self._domain_path,'Faces_Cell_Indexes')

    @property
    def Cells_Face_and_Orientation_Info(self):
        return self.domain_table(self._domain_path, 'Cells_Face_and_Orientation_Info')

    @property
    def Cells_Face_and_Orientation_Values(self):
        return self.domain_table(self._domain_path,'Cells_Face_and_Orientation_Values')

    @property
    def Faces_Area_Elevation_Info(self):
        return self.domain_table(self._domain_path,'Faces_Area_Elevation_Info')

    @property
    def Faces_Area_Elevation_Values(self):
        return self.domain_table(self._domain_path,'Faces_Area_Elevation_Values')

    @property
    def Faces_NormalUnitVector_and_Length(self):
        return self.domain_table(self._domain_path, 'Faces_NormalUnitVector_and_Length')

    @property
    def FacePoints_Coordinate(self):
        return self.domain_table(self._domain_path,'FacePoints_Coordinate')

#--------------------------------------------------------------------Functions
"""
def hdfData(hdfFile:str, path:str):
    '''Get Data from HDF'''
    with h5py.File(hdfFile,'r') as hf:
        binary_data = np.array(hf[path])
        return binary_data
"""  


#--------2D AREA GEOMETRY CALCULATIONS----------#
def fpoints_gdf(FacePoints_Coordinate):
    '''Make a geodataframe of facepoints'''
    x = [x[0] for x in FacePoints_Coordinate]
    y = [x[1] for x in FacePoints_Coordinate]
    return gpd.GeoDataFrame(gpd.GeoSeries(map(Point, zip(x, y))))

def buffer_fpoints(FacePoints_Coordinate):
    '''Buffer facepoints to intersect with breakline points'''
    facePoints=[]
    for xy in np.arange(0, FacePoints_Coordinate.shape[0]):
        facePoints.append(Point(FacePoints_Coordinate[xy]).buffer(0.1))
    return gpd.GeoDataFrame(geometry=facePoints)

def get_bline_fpoint_pairs(bLine, bLinePointCoords, gdf):
    '''Create a line object to identify facepoints that intersect
       !--> WARNING: ccurrently takes stop_point, need to pass all points in line
    '''
    print('Warning: Check this function. If needed, update to read in all line points')
    start_point = bLinePointCoords[bLine][0]
    stop_point = bLinePointCoords[bLine][-1]
#     line = LineString([start_point,stop_point])
    line = LineString(bLinePointCoords[bLine])
    print()
    fpointsOnLineGDF = gdf[gdf.intersects(line)].copy()
    return start_point, fpointsOnLineGDF

def ordered_points(fpointsOnLineGDF, fpointsGdf, start_point):
    '''Sort line vertices to establish connectivity of cells'''
    fpointsOnLine={}
    for idx in fpointsOnLineGDF.index:
        #print(idx, fpointsGdf.loc[idx])
        fpointsOnLine[idx] =fpointsGdf.loc[idx]

    sortPoints = dict()
    for idx, fp in fpointsOnLine.items():
        a = Point(start_point)
        b = fp[0]
        distance_from_start =  a.distance(b)
        sortPoints[idx] = distance_from_start

    sortedPoints = OrderedDict(sorted(sortPoints.items(), key=lambda x: x[1]))

    return sortedPoints

def get_point_combos(sortedPointList):
    '''Check this function....can't remember why this is needed...'''
    combos = []
    for i, p in enumerate(sortedPointList):
        if len(sortedPointList) > i+1:
            fpA = np.array(sortedPointList[i])
            fpB = np.array(sortedPointList[i+1])
            combos.append(np.array([fpA, fpB]))
    return combos

def build_geometry(pointCombos, Faces_FacePoint_Indexes, Cells_Minimum_Elevation, FacePoints_Face_and_Orientation_Info,
             FacePoints_Face_and_Orientation_Values, Faces_Cell_Indexes, verbose=False):
    searchTable = Faces_FacePoint_Indexes
    cells = []
    faceCellCombos = {}
    faceNormal = {}
    if verbose:
        print('Face Points','\t', 'Face','\t', 'Common Cells', '\t','aCell (min el) bCell (min el)', '\tOrthogonal Dir')
    for combo in pointCombos:
        aFaces = np.where(searchTable == combo[0])[0]
        bFaces = np.where(searchTable == combo[1])[0]
        commonFace = [f for f in aFaces if f in bFaces][0]

        faceCells = Faces_Cell_Indexes[commonFace]
        for c in faceCells: cells.append(c)

        aCellMinElev = Cells_Minimum_Elevation[faceCells[0]]
        bCellMinElev = Cells_Minimum_Elevation[faceCells[1]]


        # Read position info from d11
        startFind = FacePoints_Face_and_Orientation_Info[combo[0]][0]
        stopFind = startFind + FacePoints_Face_and_Orientation_Info[combo[0]][1]
        startFind, stopFind

        fPointFaces = FacePoints_Face_and_Orientation_Values[startFind:stopFind]
        idx = np.argwhere(fPointFaces==commonFace)
        orthoDir = fPointFaces[idx[0][0]][-1]

        faceCellCombos[commonFace] = faceCells
        faceNormal[commonFace] = orthoDir

        if verbose: print(combo,'\t', commonFace, '\t',faceCells,'\t', aCellMinElev,'\t', bCellMinElev, '\t', orthoDir)
    return cells, faceCellCombos, faceNormal

def get_crossCelll_pairs(Cells_Face_and_Orientation_Info, faceCellCombos, cells, verbose=False):
    '''Return Cells that are intersected by a known face'''
    crossCellPairs=[]
    for combo in faceCellCombos:
        crossCells = str(faceCellCombos[combo][0]) +' '+ str(faceCellCombos[combo][1])
        crossCellPairs.append(crossCells)
    if verbose: print(crossCellPairs)

    infoTable = {}
    for c in cells:
        infoTable[c] = Cells_Face_and_Orientation_Info[c]

    if verbose: print(infoTable)

    return crossCellPairs, infoTable

def get_checkCells(infoTable,Cells_Face_and_Orientation_Values, verbose=False):
    '''Check this function....can't remember why this is needed/What is computed...'''
    checkCells={}
    for cell, vals in infoTable.items():
        startFind = infoTable[cell][0]
        stopFind = startFind + infoTable[cell][1]
        cFaces = Cells_Face_and_Orientation_Values[startFind:stopFind]
        faces = [c[0] for c in cFaces]
        checkCells[cell] = faces
        if verbose: print(cell, faces)
    return checkCells

def get_sharedPairs(checkCells, crossCellPairs):
    '''Check this function....can't remember why this is needed/What is computed...'''
    checkedCombos = {}
    for i, c in enumerate(checkCells):
        for i2, c2 in enumerate(checkCells):
            combo = '{} {}'.format(c, c2)
            comboT= '{} {}'.format(c2, c)
            if (combo!=comboT) and (combo not in crossCellPairs) and (comboT not in crossCellPairs) and (combo not in checkedCombos) and (comboT not in checkedCombos):
                a = checkCells[c]
                b = checkCells[c2]
                NsharedFaces = len([f for f in a if f in b])
                if NsharedFaces !=0:
                    checkedCombos[combo] = NsharedFaces
                #print('Check {} against {}'.format(c,c2))

    return [[x.split(' ')[0], x.split(' ')[1]] for x in list(checkedCombos.keys())]

def get_groups(sharePair: list):
    '''Given all cells along breakline, group those on either side of the breakline'''
    spairs = sharePair.copy()
    g1 = spairs[0]
    spairs.remove(spairs[0])

    for i in range(3):
        for p in spairs:
            if i%2==0:
                a, b = p[0], p[1]
            else:
                a, b = p[1], p[0]

            if (a in g1):
                g1.append(b)
                spairs.remove(p)

    g1 = list(set(g1))
    g2 =  list(set([item for sublist in spairs for item in sublist]))

    return g1, g2

def map_cellFaces(faceCellCombos, group):
    '''Group Cells and faces for flow computation'''
    cellFaceMap={}
    for g in group:
        cell = int(g)
        for face, cells in faceCellCombos.items():
            if cell in cells:
                cellFaceMap[cell] = face
    return cellFaceMap


#--------FLOW CALCULATIONS----------#
def mapTable(idxs, infoArray):
    '''Helper funciton. From Info, get position of indexed data to build table'''
    table_map = {}
    for idx in idxs:
        start = infoArray[idx,0]
        stop = start + infoArray[idx,1]
        table_map[idx] = (start, stop)
    return table_map

def interp_areaEl(wsel, df):
    '''Interpolation function'''
    return np.interp(wsel,  df['Elevation'], df['Area'])

def extrapElevAreas(table_map, faceAreaElevations, faceNormalVectors,
                    extend_to=1000, plot=False):
    '''Extrapolate rating curve above lowest point on cell face line'''
    areaCurves = {}
    for face, idxs in table_map.items():

        fAEL_slice = faceAreaElevations[idxs[0]:idxs[1],[0,1]]
        faceVector = faceNormalVectors[face,2]
        maxElev = fAEL_slice[:, 0].max()
        maxArea = fAEL_slice[:, 1].max()

        # plot original curves
        if plot:
            fig, ax = plt.subplots(1,2, figsize=(20,3))
            ax[0].plot(fAEL_slice[:,1],fAEL_slice[:,0])
            ax[0].grid();ax[0].set_title(face);ax[0].set_xlabel('Area (sq ft)');ax[0].set_ylabel('WSE (ft)')

        # intialize data for extrapolation
        newElev = maxElev
        newArea = maxArea
        checkData=[]

        # extrapolate data
        for i in np.arange(1, extend_to, 1):
            newElev+=1
            newArea = maxArea + (i)*faceVector
            checkData.append([newElev, newArea])

        # append extrapolated rows to original table
        fAEL_slice = np.append(fAEL_slice, checkData, axis=0)

        # plot extrapolated curves
        if plot:
            ax[1].plot(fAEL_slice[:,1],fAEL_slice[:,0])
            ax[1].grid();ax[1].set_title(face);ax[1].set_xlabel('Area (extended, sq ft)');ax[1].set_ylabel('WSE (ft)')
        areaCurves[face] = fAEL_slice

    return areaCurves


def compute_blineFlow(breakline, wses, cellIDs, cellFaceMap, areaElevCurves, fVel, faceNormal):
    breaklineFlow = np.zeros(wses[:, cellIDs[0]].shape)

    for cell, face in cellFaceMap.items():
        wse_slice = wses[:, cell]
        faceArea = np.interp(wse_slice,  areaElevCurves[face][:,0], areaElevCurves[face][:,1])
        faceVelocity = fVel[:, face]
        faceFlow = faceVelocity*faceArea*faceNormal[face]
        breaklineFlow+=faceFlow

    return breaklineFlow


#-----------PLOT FLOW AT BREAKLINES----------------#
def singlePlot(breakline, array):
    idx = np.arange(array.shape[0])
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(idx, array);ax.grid()
    ax.set_title('{} \nMax {}, Min {}'.format(breakline, str(array.max().round(4)), str(array.min().round(4))))
    return fig, ax

def doublePlot(breakline, array1, array2):
    idx = np.arange(array1.shape[0])
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(idx, array1)
    ax.plot(idx, array2)
    a1max = str(array1.max().round(2))
    a1min = str(array1.min().round(2))
    a2max = str(array2.max().round(2))
    a2min = str(array2.min().round(2))
    ax.set_title('{} \nOption 1: Max {},  Min {}\nOption 2: Max {},  Min {}'.format(breakline,a1max ,a1min,a2max,a2min ))
    ax.grid()
    return fig, ax