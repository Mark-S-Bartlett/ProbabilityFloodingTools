import os
import time
import shutil
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


#---------------------------------------------------------------------------#

'''Functions called by Calc_Reduced_Excess_Rainfall.ipynb, which calculates 
   the reduced excess rainfall using the excess rainfall data contained
   within the passed DSS file and a randomly generated stormwater removal
   rate and capacity for each event.
'''

#---------------------------------------------------------------------------#


def read_dss_file_to_csv(dir_root: str, dir_data: str, file: str, 
    dss_util_path: str='../bin/DSSUTL.EXE', remove_temp_files: bool=True, 
                                                var: str='PRECIP') -> str:
    '''Moves the DSS utility to the data directory, reads the specified DSS 
       file, and then saves the specified variable data to a text file. 
    '''
    shutil.copy(dss_util_path, dir_data)
    os.chdir(dir_data)
    with open('FromDSS.input','w') as f: 
        f.write('WR TO={} C={}'.format( f'{var}_{file}.txt', var))
    os.system("{} {}.dss INPUT={}".format('DSSUTL.EXE',file,'FromDSS.input'))
    time.sleep(5)
    if remove_temp_files:
        os.remove(os.path.join(dir_data,'DSSUTL.EXE' ))
        os.remove(os.path.join(dir_data,'FromDSS.input' ))
    try: 
        os.chdir("..")
        os.remove(os.path.join(os.getcwd(),'c'))
    except:
        None
    output_file = os.path.join(dir_data, f'{var}_{file}.txt')
    os.chdir(dir_root)
    print('DSS File written to {}'.format(output_file))
    return output_file


def dss_data_to_table(dss_data_file: str, dss_area_key: str='B', 
                                            dss_event_key: str='F') -> list:
    '''Converts the text file containing the variable data (precipitation,
       ect.) of interest at the specified path to a list of dataframes.
    '''
    with open(dss_data_file,'r') as f:
        dss_dfs_dict={}
        for line in f.readlines():
            if '/' in line:
                dss_parts = split_dss_line_paths(line)
                event = dss_parts[dss_event_key]
                area = dss_parts[dss_area_key]
            elif 'Ver:' in line:
                continue
            elif 'Start' in line:
                d=get_dss_tseries(line)
                start=datetime.strptime(d['sdate']+d['stime'], '%d%b%Y%H%M')
                if int(d['etime'])==2400: 
                    end=datetime.strptime(d['edate']+'0000', '%d%b%Y%H%M')
                    end+=timedelta(days=1)
                else:
                    end=datetime.strptime(d['edate']+d['etime'], '%d%b%Y%H%M')                
                event_datetime_index=pd.date_range(start=start, 
                                            end=end, periods=int(d['ords']))
            elif 'Units' in line:
                continue # Placeholder in case units are needed
            else:  
                data_line_ordinates = get_ordinates(line)
                if area not in dss_dfs_dict:
                    dss_dfs_dict[area] = {}
                    dss_dfs_dict[area]['idx'] = event_datetime_index
                    dss_dfs_dict[area][event]=[]
                    for ordinate in data_line_ordinates:
                        dss_dfs_dict[area][event].append(ordinate)
                elif event not in dss_dfs_dict[area]:
                    dss_dfs_dict[area][event]=[]
                    for ordinate in data_line_ordinates:
                        dss_dfs_dict[area][event].append(ordinate)
                elif 'END' in line:
                    continue
                else:
                    for ordinate in data_line_ordinates:
                        dss_dfs_dict[area][event].append(ordinate) 
    dss_df_list=[]
    dic_keys=list(dss_dfs_dict.keys())
    for dss_dict in dss_dfs_dict.keys():
        df = pd.DataFrame.from_dict(dss_dfs_dict[dss_dict], orient='index').T
        df.set_index('idx', inplace=True)
        df.index.name=None
        df.replace(np.nan,0, inplace=True)
        df.name=dss_dict
        dss_df_list.append(df)                
    print(f'{len(dss_dfs_dict[area].keys())-1} Runs in DSS')
    print(f'{len(dic_keys)} Areas in DSS with names: {dic_keys}')
    return dss_df_list


def split_dss_line_paths(line: str) -> dict:
    '''Splits the passed string using the forward slash.
    '''
    dss_parts=line.split('/')
    dic={'A':dss_parts[1], 'B':dss_parts[2], 'C':dss_parts[3],
            'D':dss_parts[4], 'E':dss_parts[5], 'F':dss_parts[6]}
    return dic


def get_dss_tseries(line: str) -> dict:
    '''Splits the passed string using the semicolon in order to return the 
       start date, start time, end date, end time, and ordinates.
    '''
    dss_parts=line.split(';')
    dic={'sdate':dss_parts[0].replace('at','').replace('Start:','').replace('hours','').split()[0],
        'stime':dss_parts[0].replace('at','').replace('Start:','').replace('hours','').split()[1],
        'edate':dss_parts[1].replace('at','').replace('End:','').replace('hours','').split()[0],
        'etime':dss_parts[1].replace('at','').replace('End:','').replace('hours','').split()[1],
        'ords':dss_parts[2].replace('Number:','').replace('\n','').strip()}  
    return dic   


def get_ordinates(line: str) -> list:
    '''From the passed line, extract the variable data (precipitation, ect.). 
    '''
    tseries_ordinates=[]
    field_width = 10 
    for i in range(0, 7):
        if 'END' in line:
            continue
        else:
            try: #All lines may not have 7 ordinates
                ordinate = line[i*field_width:field_width+field_width*i]
                tseries_ordinates.append(float(ordinate))
            except:
                return tseries_ordinates
    return tseries_ordinates


def make_rainfall_adjustment(df: pd.DataFrame, col: pd.Series, ts: int, 
						minrate: float, maxrate: float, min_cap: float=1.0, 
												max_cap: float=1.0) -> list:
    '''Randomly selects a stormwater removal rate, calculates the design 
       capacity, and uses the adjust_excess function to calculate the reduced
       excess rainfall for each event within the passed dataframe.
    '''
    minrate30 = minrate*(ts/30.0)
    maxrate30 = maxrate*(ts/30.0)
    adj_rate=np.random.uniform(minrate30, maxrate30)   
    max_cap=np.random.uniform(min_cap, max_cap)*(adj_rate*(60.0/ts*24.0))    
    adj_excess = adjust_excess(df, col, adj_rate, max_cap)
    results=[adj_rate, max_cap, adj_excess]
    return results


def adjust_excess(df: pd.DataFrame, col: pd.Series, adj_rate: float, 
                                        max_capacity: float) -> pd.DataFrame:
    '''Given the stormwater removal rate and the design capacity, the 
       reduced excess rainfall (runoff) is calculated for the event specified
       by the passed column.
    '''
    adjusted = df[col] - adj_rate                               
    adjusted[adjusted <  0] = 0                                  
    capacity = 0                                                    
    df_adj = pd.DataFrame(adjusted)                                    
    for i in df_adj.index:
        capacity+= df_adj[col].loc[i]                          
        if capacity >= max_capacity:                             
             df_adj[df_adj.columns[0]].loc[i] = df[col].loc[i]       
    return df_adj


def plot_reduced_excess(df: pd.DataFrame, df2: pd.DataFrame, col: pd.Series,
                        tstep: int, duration: int, adjustment_rate: float,
                                        max_capacity: float) -> plt.subplots:
    '''Plots the incremental excess rainfall and adjusted excess rainfall in
       the same figure.
    '''
    fig, ax = plt.subplots(figsize=(24,3))
    xmax=int(duration*(60/tstep))
    ax.plot(df[0:xmax].index, df[0:xmax][col], '--', color='grey', 
                                    label = 'Incremental Excess Rainfall')
    ax.set_ylabel('Inches')
    ax.set_xlabel('Hours')
    ax.plot(df2[0:xmax].index, df2[0:xmax][col], '-', color = 'blue', 
        label = 'Adjusted Excess Rainfall:\n{} inches/hour'.format(round(adjustment_rate, 5)))
    ax.set_title('Cumulative Excess Rainfall {} inches\n Design Capacity {} inches'.format(np.round(df[col].sum(),2),np.round(max_capacity,2)))
    ax.grid()
    ax.legend()


def dss_input_data(data_dir: str, var, units: str='INCHES', ts: str='15MIN', 
                            dtype: str='INST-VAL', IMP: str='DSS_MAP.input', 
                                        to_dss: str='ToDSS.input') -> None:
    '''Creates an input file containing the data structure, aka the map,
       for DSSUTL.EXE.
    '''
    var8=var[:8]
    output_file = os.path.join(data_dir, IMP)
    datastring = "EV {0}=///{0}//{1}// UNITS={2} TYPE={3}\nEF [APART] [BPART] [DATE] [TIME] [{0}]\nIMP {4}".format(var8, ts, units, dtype, to_dss)
    with open(output_file, 'w') as f: 
        f.write(datastring)
    return None


def precip_df_to_dss(df: pd.DataFrame, dir_data: str, area: str, 
                        duration: int,  temp_file: str='ToDSS.input', 
                        dss_util_path: str= '../../bin/DSSUTL.EXE') -> None:
    '''Adds the reduced excess rainfall data to an input file according to
       the struture specified within DSS_MAP.input.
    '''
    temp_data_file = os.path.join(dir_data, temp_file)
    cols = df.columns.tolist()
    with open(temp_data_file, 'w') as f:
        for i, col in enumerate(cols):
            m_dtm = datetime(2009,5, 1)
            event_data = df[col]
            for t in event_data.index:
                m_dtm+=pd.Timedelta(hours=0.25)
                htime_string = datetime.strftime(m_dtm, '%d%b%Y %H%M')
                if htime_string[10:] == '0000':
                    hec_time=m_dtm-pd.Timedelta(hours=24)
                    hec_time=datetime.strftime(hec_time, '%d%b%Y %H%M')
                    htime_string = hec_time.replace('0000', '2400')
                precip = event_data.loc[t]
                f.write('"{}"'.format(area)+' '+col+' '+htime_string+' '+str(precip)+'\n')
    return None


def make_dss_file(dir_data: str, filename_dss: str, dir_root: str, 
                            dss_util_path: str= '../bin/DSSUTL.EXE', 
                                    remove_temp_files: bool=True) -> None:
    '''Runs the DSSUTL executable using the DSS_MAP.input file to map the 
       reduced excess rainfall data from the ToDSS.input file and saves the
       results to a dss file.
    '''
    shutil.copy(dss_util_path, dir_data)
    os.chdir(dir_data)
    os.system("{0} {1}.dss INPUT={2}".format('DSSUTL.EXE', filename_dss,'DSS_MAP.input'))
    time.sleep(5)
    if remove_temp_files:
        os.remove(os.path.join(dir_data,'DSS_MAP.input' ))
        os.remove(os.path.join(dir_data,'DSSUTL.EXE' ))
        os.remove(os.path.join(dir_data,'ToDSS.input' ))
    os.chdir(dir_root)
    filepath=os.path.join(dir_data, filename_dss)
    print('Dss File written to {0}.dss'.format(filepath))
    return None


    #---------------------------------------------------------------------------#