import os
import pathlib
import folium
from folium.folium import Map
import branca
from glob import glob
import pandas as pd
import geopandas as gpd
from osgeo import gdal, osr
gdal.UseExceptions() 
from shapely.geometry import Point, LineString
from scipy import interpolate
import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')


def comp_msgs(hf):
    compute_data=[]
    binary_data = np.array(hf['/Results/Summary/Compute Messages (text)'])
    messages = [data.decode('UTF-8').strip() for data in binary_data]
    messages=messages[0].split('\r')
    msg = [m.strip('\n') for m in messages if (len(m) > 0) and (m.strip('\n')[0:4] != 'File')]
    return msg

def iterations_2d(hf, domain:str):
    iterations = np.array(hf[f'/Results/Unsteady/Output/Output Blocks/Computation Block/2D Flow Areas/{domain}/2D Iterations'])
    iteration_errors = np.array(hf[f'/Results/Unsteady/Output/Output Blocks/Computation Block/2D Flow Areas/{domain}/2D Iteration Error'])
    return iterations, iteration_errors    
    
def get_2d_errors(hf, domains:list):    
    domain_errors = {}
    for d in domains:
        iterations, iteration_errors  = iterations_2d(hf, d)
        domain_errors[d] = iteration_errors.sum()
        
    return domain_errors

def maxv(rasplan:str, domain:str):
    with h5py.File(rasplan,'r') as hf:
        binary_data = np.array(hf[f'/Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/{domain}/Maximum Face Velocity'])
        return binary_data

def minv(rasplan:str, domain:str):
    with h5py.File(rasplan,'r') as hf:
        binary_data = np.array(hf[f'/Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/{domain}/Minimum Face Velocity'])
        return binary_data    
    
def get_velocities(rasplan:str, domains:list):    
    domain_velocities = {}
    
    for d in domains:
        v_stats = {}
        print(d)
        
        maxvel = maxv(rasplan, d)
        minvel = minv(rasplan, d)

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
    
# RAS FUNCTIONS
def get_domain_names(hf):
    domain_paths = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/'
    domains = list(hf[domain_paths].keys())
    return domains

def boundary_conditions_domain(hf, domain:str):
    figs=[]
    df = pd.DataFrame()
    bcdata={}
    bcs = np.array(hf[f'/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{domain}/Boundary Conditions/'])
    for bc in bcs:
        bcdata[bc] = np.array(hf[f'/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{domain}/Boundary Conditions/{bc}'])
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

def get_mannings_data(rasplan:str):
    with h5py.File(rasplan,'r') as hf:
        binary_data = np.array(hf[r"/Geometry/Land Cover (Manning's n)/Calibration Table"])
        return dict((x,y) for x, y in binary_data)
    


def get_structure_names(plan:str, print_path=False):
    structs = {}
    with h5py.File(plan,'r') as hf:
        structure_paths = ['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/SA 2D Area Conn',
                           'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Lateral Structures']
        for path in structure_paths:
            try:
                structures = np.array(hf[path])
                for structure_found in structures:
                    if print_path: print(f'Structure: {path}/{structure_found}')
                    structs[structure_found] = f'{path}/{structure_found}'
            except:
                continue
    return structs

def read_plan(rasplan: str):
    with h5py.File(rasplan,'r') as hf:
        _attrs        = '/Geometry/Structures/Attributes'
        _cline_info   = '/Geometry/Structures/Centerline Info'
        _cline_parts  = '/Geometry/Structures/Centerline Parts'
        _cline_points = '/Geometry/Structures/Centerline Points'
        _profiles     = '/Geometry/Structures/Profiles'
        
        attrs         = pd.DataFrame(np.array(hf[f'{_attrs}']), dtype=str)
        cline_info    = pd.DataFrame(np.array(hf[f'{_cline_info}']), dtype=str)
        cline_info.rename(columns={0:'Start', 1:'nrows'}, inplace=True)
        cline_info    = cline_info[['Start', 'nrows']].apply(pd.to_numeric)
        
        attrs.index.name = 'StructID'
        cline_info.index.name='StructID'
        
        cline_parts  = pd.DataFrame(np.array(hf[f'{_cline_parts}']), dtype=str)
        cline_points  = pd.DataFrame(np.array(hf[f'{_cline_points}']), dtype=str)
        profiles      = pd.DataFrame(np.array(hf[f'{_profiles}']), dtype=str)
        
    return attrs, cline_info, cline_points, profiles, cline_parts

def read_plan2(rasplan: str):
    with h5py.File(rasplan,'r') as hf:
        _attrs        = '/Geometry/Structures/Centerline Info'
        data = pd.DataFrame(np.array(hf[f'{_attrs}']),dtype=str)
        data.rename(columns={0:'Start', 1:'nrows'}, inplace=True)
        data.index.name='StructID'
        
    return data[['Start', 'nrows']].apply(pd.to_numeric)

# Read Ras Geometric data at structures
def updated_structure_data(rasplan, attrs, cline_info, cline_points, profiles, cline_parts, structures):
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


def get_structure_data(plan:str, structure:str, path:str):
    
    with h5py.File(plan,'r') as hf:
        structure_data = np.array(hf[path])
        
        if 'Culvert Groups' in structure_data:
            culverts=1
            #print('Cluverts Found')
            
        _hw_tw_segments = 'HW TW Segments'  
        
        _hw_cells    = 'Headwater Cells'
        _tw_cells    = 'Tailwater Cells'            
        _hw_tw_cells = 'HW TW Cells'
        _hw_tw_station =  'HW TW Station'
        
        _hw_face_pts      = 'Headwater Face Points'
        _hw_face_pts_stns = 'Headwater Face Points Stations'
        _tw_face_pts      = 'Tailwater Face Points'
        _tw_face_pts_stns = 'Tailwater Face Points Stations'
        
        _wse_hw_cells = 'Water Surface HW Cells'
        _wse_tw_cells = 'Water Surface TW Cells'
        _wse_hw       = 'Water Surface HW'
        _wse_tw       = 'Water Surface TW'
        
        _geom_info = 'Geometric Info'
        
        if _hw_tw_segments in structure_data:
            try:
            
                hw_cells = np.array(hf[f'{path}/{_hw_tw_segments}/{_hw_cells}'])
                hw_cells = [data.decode('UTF-8').strip() for data in hw_cells]

                tw_cells = np.array(hf[f'{path}/{_hw_tw_segments}/{_tw_cells}'])
                tw_cells = [data.decode('UTF-8').strip() for data in tw_cells]

                hw_tw_stns = np.array(hf[f'{path}/{_hw_tw_segments}/{_hw_tw_station}'])
                flow = np.array(hf[f'{path}/{_hw_tw_segments}/Flow'])
            except:
                hw_cells =None # 1D XS's
                tw_cells = None

                hw_tw_stns = np.array(hf[f'{path}/{_hw_tw_segments}/{_hw_tw_station}'])
                flow = np.array(hf[f'{path}/{_hw_tw_segments}/Flow'])
                


        if _hw_tw_cells in structure_data:
            #print(path)
            wse_hw = np.array(hf[f'{path}/{_hw_tw_segments}/{_wse_hw}'])
            wse_tw = np.array(hf[f'{path}/{_hw_tw_segments}/{_wse_tw}'])
        else:
            wse_hw,wse_tw=None, None

                   
    return flow, wse_hw, wse_tw, hw_cells, tw_cells ,hw_tw_stns


def find_closest(stations:list, target):
    '''Helper function to find nearest target in station list (to keep fewer points to map)'''
    A = np.array(sorted(stations))
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def read_plot_wses(rasplan, structure, hw_face_pts_stns, wse_hw, projection):
    with h5py.File(rasplan,'r') as hf:
        _attrs        = '/Geometry/Structures/Attributes'
        _cline_parts  = '/Geometry/Structures/Centerline Parts'
        _cline_points = '/Geometry/Structures/Centerline Points'
        _profiles     = '/Geometry/Structures/Profiles'

        attrs         = pd.DataFrame(np.array(hf[f'{_attrs}']),dtype=str)
        profiles      = pd.DataFrame(np.array(hf[f'{_profiles}']),dtype=str)
        sdata         = pd.DataFrame()
        all_structs   = gpd.GeoDataFrame()

        cline_parts   = np.array(hf[f'{_cline_parts}'])
        cline_points  = np.array(hf[f'{_cline_points}'])

    profiles['Station'] = pd.to_numeric(profiles['Station'])    
    profiles['Elevation'] = pd.to_numeric(profiles['Elevation'])  
    zeros = profiles[profiles['Station']==0.0].index

    for i in attrs.index:     
        _connection ='Connection'
        connection  = attrs.loc[i][_connection]
        
        if connection in structure:
            if i==0:
                adjustment=0
            else:
                adjustment = cline_parts[i-1][1]
                
            start, stop = cline_parts[i][0]+adjustment, cline_parts[i][1]-adjustment
            connection_cline_points = cline_points[start:stop]

            if i+1  in attrs.index:
                sdata = profiles.loc[zeros[i]:zeros[i+1]-1].copy()
            else:
                sdata = profiles.loc[zeros[i]:].copy()
            
            maxels =  np.amax(wse_hw, axis=0)

    x = connection_cline_points[start:stop,0]
    y = connection_cline_points[start:stop,1]
    
    s = gpd.GeoSeries(map(Point, zip(x, y)))
    all_structs.geometry = s
    all_structs['ID'] = structure

    total_distance=0
    point1 = all_structs.loc[0, 'geometry']
    for i in all_structs.index:
        point2 = all_structs.loc[i, 'geometry']
        total_distance += point2.distance(point1)
        all_structs.loc[i, 'Station'] = total_distance
        point1 = all_structs.loc[i, 'geometry']
        
    sdata.set_index('Station', inplace=True)

    f = interpolate.interp1d(sdata.index, sdata.Elevation, fill_value='extrapolate')
    all_structs['Crest'] = all_structs['Station'].apply(f)
    
    f = interpolate.interp1d(hw_face_pts_stns, maxels, fill_value='extrapolate')
    all_structs['Max WSE'] = all_structs['Station'].apply(f) 
    all_structs.crs=projection
    

    
    if all_structs.Station.max() > 10000:
        map_interval = 1000
    elif (all_structs.Station.max() <= 1000) and (all_structs.Station.max() <= 10000):
        map_interval = 200
    else:
        map_interval = 25
        
    filtered_stations = []
    for target in np.arange(0, int(all_structs.Station.max()), map_interval):
        keep_idx = find_closest(all_structs.Station, target)
        filtered_stations.append(keep_idx)

    #return all_structs, connection_cline_points
    return all_structs.loc[filtered_stations].reset_index(), connection_cline_points

#--Folium 
# MAPPING FUNCTIONS
def get_map_center(gdf):
    #reads geodataframe, returns average of points from a line file
    lats = []
    lons = []
    
    for index, row in gdf.iterrows():
        lons.append(row['geometry'].bounds[0])
        lons.append(row['geometry'].bounds[2])
        lats.append(row['geometry'].bounds[1])
        lats.append(row['geometry'].bounds[3])
    return np.mean(lats), np.mean(lons)

def add_tiles(folmap): 
    EsriImagery = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    EsriAttribution = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
    folmap.add_tile_layer(tiles=EsriImagery, attr=EsriAttribution, name='EsriImagery')
    folium.LayerControl().add_to(folmap)
    return folmap

def map_results_raster(structure_list, raster, breach=None, book=None, width=800, height=700, zoom = 14):
    from folium import raster_layers
    colormap=lambda x: (0, 0, 0, x)
    
    overtopped={}
    # Read in Raster Data
    src = gdal.Open(raster)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    centerx, centery  = np.mean([ulx, lrx]),np.mean([uly, lry])
    img = src.ReadAsArray()
        
    colormap=lambda x: (0, 1, 1, x)
    folmap = Map(width=800, height=700, location=[centery, centerx], zoom_start=zoom)
    folmap.add_child(raster_layers.ImageOverlay(img, colormap=colormap,opacity=0.5, bounds =[[lry, lrx],[uly, ulx]], mercator_project=True))
    
    if isinstance(breach, gpd.geodataframe.GeoDataFrame):
        if breach.crs != '4326':
            breach = breach.to_crs({'init' :'epsg:4326'})
        html = f"""<h4>Breach Location</h4>"""    
        iframe=branca.element.IFrame(html=html, width=120, height=80)    
        popup = folium.Popup(iframe)  
        folmap.add_child(folium.CircleMarker(location=[breach.loc[0, 'geometry'].y, 
                                     breach.loc[0, 'geometry'].x], 
                                     popup= popup,radius=6, weight=12, color='Red'))
        
    if isinstance(book, gpd.geodataframe.GeoDataFrame):
        print('Book!', book.crs)
        if book.crs != {'init': 'epsg:4326'}:
            print('Projecting')
            book = book.to_crs({'init' :'epsg:4326'})
            # Placeholder for plotting structures
            

    for structures in structure_list:
        #print(structures.crs)
        if structures.crs != '4326':
            structures = structures.to_crs({'init' :'epsg:4326'})
        try:


            for idx in structures.index:
                sID =  structures.loc[idx, 'ID']  
                sWSE = round(structures.loc[idx, 'Max WSE'],2)
                sCrest = round(structures.loc[idx, 'Crest'],2)
                sStation = int(structures.loc[idx, 'Station'])

                if sWSE < sCrest:
                    html = f"""<h4>{sID}</h4><p>Station {sStation}</p><p>Levee Crest {sCrest}</p><p>Max WSE {sWSE}</p>"""
                    iframe=branca.element.IFrame(html=html, width=200, height=150)    
                    popup = folium.Popup(iframe)  
                    folmap.add_child(folium.CircleMarker(location=[structures.loc[idx, 'geometry'].y, 
                                                 structures.loc[idx, 'geometry'].x], 
                                                 popup= popup,radius=3, weight=3, color='black'))
                else:
                    max_depth=round(sWSE-sCrest,2)
                    overtopped[f'{sID} Station {sStation}'] =  max_depth
                    html = f"""<h4>{sID}</h4><p>Station {sStation}</p><p>Levee Crest {sCrest}</p><p>Max WSE {sWSE}</p><p>Max Overtopping Depth {max_depth}</p>"""
                    iframe=branca.element.IFrame(html=html, width=200, height=250)    
                    popup = folium.Popup(iframe) 
                    folmap.add_child(folium.CircleMarker(location=[structures.loc[idx, 'geometry'].y, 
                                                 structures.loc[idx, 'geometry'].x], 
                                                 popup= popup,radius=4, weight=4, color='red'))
                    
            line_points = zip(structures.geometry.y.values,structures.geometry.x.values)
            folium.PolyLine(list(line_points), color='black').add_to(folmap)
        except:
            print('ERROR')
                
    add_tiles(folmap)
    return folmap, overtopped

# Raster Functions

def prep_wse(wse_grid, bool_grid, nullval = -9999.0):
    '''Make Boolean'''
    ds = gdal.Open(wse_grid)
    rb = ds.GetRasterBand(1)
    xsize = rb.XSize
    ysize = rb.YSize
    ystep = int(ysize / 10)
    yresidual = ysize - (ystep * 10)
    
    #__ADDD
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(bool_grid, xsize, ysize, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform(ds.GetGeoTransform())
    outband = outRaster.GetRasterBand(1)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
    
    for i in range(10):
        if i != 9:
            chunk = rb.ReadAsArray(0, ystep * i, xsize, ystep)
            chunk_bool = chunk !=nullval
            outband.WriteArray(chunk_bool, 0, ystep * i, gdal.GDT_Byte)

        else:
            chunk = rb.ReadAsArray(0, ystep * i, xsize, ystep + yresidual)
            chunk_bool = chunk !=nullval
            outband.WriteArray(chunk_bool, 0, ystep * i, gdal.GDT_Byte)

    ds = None
    return None

def coarsen_raster(model_path, wsegrid, BruteForce_projection=None):
            # Reproject Raster
    input_raster = gdal.Open(wsegrid)
    prj_raster = os.path.join(model_path,'prj_wse.tif')
    if BruteForce_projection:
        gdal.Warp(prj_raster, input_raster,srcSRS=BruteForce_projection, dstSRS='EPSG:4326')
    else:
        gdal.Warp(prj_raster, input_raster, dstSRS='EPSG:4326') # Case: Raster has projection in metadata

    # Make Bool
    bool_raster = os.path.join(model_path,'bool.tif')
    prep_wse(prj_raster, bool_raster)

    # Reproject Raster
    input_raster = gdal.Open(bool_raster)
    nb_raster = os.path.join(model_path,'nbraster.tif')
    gdal.Translate(nb_raster, input_raster,format='GTiff', width=2000, height=2000, resampleAlg='average')
    return bool_raster, nb_raster 

'''
def map_results_raster(structure_list, raster, width=800, height=700, zoom = 14):
    from folium import raster_layers
    colormap=lambda x: (0, 0, 0, x)
    
    overtopped={}
    # Read in Raster Data
    src = gdal.Open(raster)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    centerx, centery  = np.mean([ulx, lrx]),np.mean([uly, lry])
    img = src.ReadAsArray()
        
    colormap=lambda x: (0, 1, 1, x)
    folmap = Map(width=800, height=700, location=[centery, centerx], zoom_start=8)
    folmap.add_child(raster_layers.ImageOverlay(img, colormap=colormap,opacity=0.5, bounds =[[lry, lrx],[uly, ulx]], mercator_project=True))

    for structures in structure_list:
        #print(structures.crs)
        if structures.crs != '4326':
            structures = structures.to_crs({'init' :'epsg:4326'})
        try:


            for idx in structures.index:
                sID =  structures.loc[idx, 'ID']  
                sWSE = round(structures.loc[idx, 'Max WSE'],2)
                sCrest = round(structures.loc[idx, 'Crest'],2)
                sStation = int(structures.loc[idx, 'Station'])

                if sWSE < sCrest:
                    html = f"""<h4>{sID}</h4><p>Station {sStation}</p><p>Levee Crest {sCrest}</p><p>Max WSE {sWSE}</p>"""
                    iframe=branca.element.IFrame(html=html, width=200, height=150)    
                    popup = folium.Popup(iframe)  
                    folmap.add_child(folium.CircleMarker(location=[structures.loc[idx, 'geometry'].y, 
                                                 structures.loc[idx, 'geometry'].x], 
                                                 popup= popup,radius=4, weight=4, color='black'))
                else:
                    max_depth=round(sWSE-sCrest,2)
                    overtopped[f'{sID} Station {sStation}'] =  max_depth
                    html = f"""<h4>{sID}</h4><p>Station {sStation}</p><p>Levee Crest {sCrest}</p><p>Max WSE {sWSE}</p><p>Max Overtopping Depth {max_depth}</p>"""
                    iframe=branca.element.IFrame(html=html, width=200, height=250)    
                    popup = folium.Popup(iframe) 
                    folmap.add_child(folium.CircleMarker(location=[structures.loc[idx, 'geometry'].y, 
                                                 structures.loc[idx, 'geometry'].x], 
                                                 popup= popup,radius=6, weight=6, color='red'))
                    
            line_points = zip(structures.geometry.y.values,structures.geometry.x.values)
            folium.PolyLine(list(line_points), color='black').add_to(folmap)
        except:
            print('ERROR')
                
    add_tiles(folmap)
    return folmap, overtopped
'''

# GDAL Functions            
def query(x,y, gt, rb):
    px = int((x - gt[0]) / gt[1])
    py = int((y - gt[3]) / gt[5])
    return rb.ReadAsArray(px,py,1,1)[0][0]
    
def get_structure_wses_from_grid(gdf, wse_grid, projection):
    if gdf.crs != projection:
        gdf = gdf.to_crs(projection)
    
    points = (gdf.geometry.x, gdf.geometry.y)
    src = gdal.Open(wse_grid)
    gt = src.GetGeoTransform()
    rb = src.GetRasterBand(1)
    results = {}
    for i, point in enumerate(points[0]):
        lon, lat = points[0][i], points[1][i]
        value = query(lon,lat, gt, rb)
        results[i] = value
    return pd.Series(results)     