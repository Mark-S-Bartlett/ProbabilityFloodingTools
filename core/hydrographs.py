import os
import csv
import json
import urllib3
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import IFrame
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters

'''

Under development

'''


def open_station_website(USGS_ID:str) -> IFrame:
    '''Open the selected USGS station website. Users can use this website to prepare the required parameters for retrieving the desired records.'''
    
    return IFrame(src="https://waterdata.usgs.gov/usa/nwis/uv?{}".format(USGS_ID), width='100%', height='500px')

 
def usgs_data(station_id:str, start_date:str, end_date:str, parameter:str) -> dict:
    '''Retrieve time series data for a USGS gauge.'''
        
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    http = urllib3.PoolManager()

    # Define URL parameters    
    data_format = "json"
    station_id = station_id
    start_date = start_date
    end_date = end_date
    parameter = parameter
    site_type = "ST"
    site_status = "all"
    
    # Build the URL to retrieve the data
    st_service_url = "https://nwis.waterservices.usgs.gov/nwis/iv/?format="+ data_format +\
    "&sites="+ station_id +"&startDT="+ start_date +"&endDT="+ end_date +"&parameterCd="+ parameter +\
    "&siteType="+ site_type +"&siteStatus="+ site_status
    url = http.request('GET', st_service_url).data
    response = json.loads(url)
    usgs_values = response['value']['timeSeries'][0]['values'][0]['value']
    
    # usgs_values = {'value': value in str, 'qualifiers': qualifiers in str, 'dateTime': dateTime in str}
    return usgs_values


def usgs_to_sliced_df(usgs_values:dict, data_threshold:float) -> (pd.DataFrame, pd.DataFrame):
    '''Return two dataframes: 
        1) dataframe for the whole retrieved data (df).
        2) dataframe for the sliced data based on a defined threshold (dfslice).'''
    
    df = pd.DataFrame.from_records(usgs_values)
    # Exclude any row with a missing value 
    df = df[df.value != '-999999']
    # Convert the timezone to UTC
    df['dateTime'] =  pd.to_datetime(df['dateTime'], utc=True)
    df['flow']= pd.to_numeric(df['value'])
    df = df.assign(increment_hr = df.dateTime.diff())
    dfslice = df[df['flow']> data_threshold].copy()
    
    return df, dfslice


def events_dic(dataframe:pd.DataFrame) -> dict:
    '''Return a dictionary with chunked storm events based on the sliced dataframe index. 
    This is to identify each storm event individually.'''
    
        
    events = {}
    chunks = []

    j=0
    for i, idx in enumerate(dataframe.index):
        if idx < dataframe.index[-1]:
            idx, idxnext = dataframe.index[i],  dataframe.index[i+1]
            if idxnext - idx > 1:
                events[j] = chunks
                chunks = []
                j+=1
            else:
                chunks.append(idx)
    events[j+1] = chunks
    
    # events = {event counter: dataframe index }
    return events

 
def tmp_df(tmp:pd.DataFrame, v:list) -> (pd.DataFrame, float, interp1d):
    '''tmp: new created dataframe extracted from the "dfslice" dataframe for each event.
    v: the key for each event in the events dictionary.
    
    Return:
        1) A Built and enhanced temporary dataframe for each selected event to use it for the analysis.
        2) the full duration of the storm event in hours.
        3) the interpolation function that fits the hydrograph of the event.'''
    
    tmp = tmp.assign(cumulative_hr = tmp.loc[v[0], 'increment_hr'] - tmp.loc[v[0], 'increment_hr'])
    for i in v[1:]:
        tmp.loc[i, 'cumulative_hr'] = tmp.loc[i, 'increment_hr'] + tmp.loc[i-1, 'cumulative_hr']
    tmp['cumulative_hr'] = pd.to_timedelta(tmp['cumulative_hr'])
    tmp['cumulative_hours'] = tmp['cumulative_hr'].dt.days * 24 + tmp['cumulative_hr'].dt.seconds /3600    
    cum_hr_max = tmp['cumulative_hours'].max()
    f = interp1d(tmp['cumulative_hours'], tmp['flow'], kind='cubic', fill_value="extrapolate")
    return tmp, cum_hr_max, f
    
               
def bin_avg_cum_hr(bin_sv:float, bin_ev:float, dfslice:pd.DataFrame, bins_avg_cum_hr:dict, events:dict, excluded_dates: list) -> dict:
    '''Return a dictionary with the value of the average maximum duration for each bin.
    This is a prelminiary step to generate the average hydrograph for each bin.'''
    
    all_cum_hr_max = []
    for k, v in events.items():
        tmp = dfslice.loc[v].copy()
        if tmp.shape[0] > 5:
            tmp, cum_hr_max, f = tmp_df(tmp, v)
            if str(tmp.loc[tmp['flow'] == tmp['flow'].max(), 'dateTime'].values[0]).split('T')[0] not in excluded_dates:
                if bin_sv < tmp.cumulative_hours.max() <= bin_ev:
                    all_cum_hr_max.append(cum_hr_max)
    bins_avg_cum_hr.update( {bin_ev : statistics.mean(all_cum_hr_max)} )
    
    
def bins_avg_cum_hr_dic(selected_bins:list, dfslice, events:dict, excluded_dates: list = None) -> dict:
    '''Return a dictionary with all bins duration.'''
    
    if excluded_dates==None:
        excluded_dates = []
    bins_avg_cum_hr = {}    
    for i in range(len(selected_bins)-1):
        bin_avg_cum_hr(selected_bins[i], selected_bins[i+1], dfslice, bins_avg_cum_hr, events, excluded_dates)
        
    # bins_avg_cum_hr = {bin range end value: average of bin storms duration}
    return bins_avg_cum_hr
    
    
def hydrograph_generation(bin_sv:float, bin_ev:float, int_increment:float, dfslice:pd.DataFrame, bins_avg_cum_hr:dict, events:dict, excluded_dates:list, data_threshold:int) -> (np.array, np.array):
    '''Return a generated average hydrograph by averaging each bin hydrographs.'''
    
    flow_df_normailzed = pd.DataFrame()
    for k, v in events.items():
        tmp = dfslice.loc[v].copy()
        if tmp.shape[0] > 5:
            tmp, cum_hr_max, f = tmp_df(tmp, v)
            date_peak_max = str(tmp.loc[tmp['flow'] == tmp['flow'].max(), 'dateTime'].values[0]).split('T')[0]
            if date_peak_max not in excluded_dates:
                if bin_sv < tmp.cumulative_hours.max() <= bin_ev:
                    x_f = np.append(np.arange(0,cum_hr_max, cum_hr_max/int_increment),cum_hr_max)
                    y_f = f(x_f)
                    x_f_min_max = x_f/x_f.max() 
                    y_f_min_max = (y_f-data_threshold)/(y_f.max()-data_threshold) #(y_f - y_f.min())/(y_f.max()) - y_f.min())
                    flow_df_normailzed[date_peak_max] = y_f_min_max


    flow_df_normailzed.iloc[0] = 0.0
    flow_df_normailzed.iloc[-1] = 0.0

    flow_df_normailzed['mean'] = flow_df_normailzed.mean(axis=1)
    flow_df_normailzed['mean'] = flow_df_normailzed['mean']/flow_df_normailzed['mean'].max()
    flow_df_normailzed['Time'] = bins_avg_cum_hr[bin_ev] * x_f_min_max
    
    return flow_df_normailzed['Time'], flow_df_normailzed['mean'], flow_df_normailzed
    
    
def generate_q_mean_json(out_file_name:str, q_mean:pd.DataFrame, boundary_name:str, timestep:float, mean_hydrograph_x:dict, mean_hydrograph_y:dict) -> json.dump:
    '''Generate a JSON file that includes the created hydrographs time series'''

    boundary_name_dic ={}
    boundary_data_dic = {}
    mean_hydrograph_dic = {}
    for yr, data in mean_hydrograph_x.items(): 
                event_hydrograph_dic = {}
                event_name = q_mean.index[q_mean['RI'] == yr].tolist()[0]
                f_q_mean = interp1d(mean_hydrograph_x[yr], mean_hydrograph_y[yr], kind='cubic', fill_value="extrapolate")
                x_q_mean = np.arange(mean_hydrograph_x[yr].min(),mean_hydrograph_x[yr].max(), timestep/60.0)
                y_q_mean = f_q_mean(x_q_mean)
                for i in range(len(x_q_mean)):
                    event_hydrograph_dic.update({x_q_mean[i] : round(y_q_mean[i],2)})
                mean_hydrograph_dic.update({event_name:event_hydrograph_dic})

    boundary_data_dic.update({boundary_name:mean_hydrograph_dic})
    boundary_name_dic.update({"BCName":boundary_data_dic})

    with open(out_file_name, 'w') as fp:
        json.dump(boundary_name_dic, fp)

        
        
def save_plot_preprocessing(data_threshold:str, selected_bins:list, selected_peaks:dict, dfslice:pd.DataFrame, q_mean:pd.DataFrame, bins_avg_cum_hr:dict, events:dict, int_increment:float, excluded_dates:list) -> (dict, dict, dict):
    '''Return:
        1) a dictionary with each bin return periods.
        2) mean hydrograhs x and y values.'''
    
    mean_hydrograph_x ={}
    mean_hydrograph_y ={}
    bin_return_pd = {}
    for i in range(len(selected_bins)-1):
        bin_avg_cum_hr(selected_bins[i], selected_bins[i+1], dfslice, bins_avg_cum_hr, events, excluded_dates)
        time, flow_normalized, flow_df_normailzed = hydrograph_generation(selected_bins[i], selected_bins[i+1], int_increment, dfslice, bins_avg_cum_hr, events, excluded_dates, data_threshold)
        bin_return_pd_list = []
        for val in q_mean.Q_Mean_cfs.values:        
            return_pd = list(q_mean.loc[q_mean['Q_Mean_cfs'] == val, 'RI'])
            for rp in return_pd:
                if val <= selected_peaks[selected_bins[i+1]] and float(rp) not in mean_hydrograph_x.keys():            
                    mean_hydrograph_x.update({float(rp) : time})
                    flow_generated = data_threshold + (val - data_threshold) * flow_normalized
                    #flow_generated = val*flow_normalized               
                    mean_hydrograph_y.update({float(rp) : flow_generated})
                    bin_return_pd_list.append(float(rp))
        bin_return_pd.update({selected_bins[i+1] : bin_return_pd_list})
    
    # bin_return_pd = {bin range end value: list of return periods within the bin range}
    # mean_hydrograph_x = {return period : hydrograph duration in hours}
    # mean_hydrograph_y = {return period : hydrograph flow value in CFS}
    return bin_return_pd, mean_hydrograph_x, mean_hydrograph_y, flow_df_normailzed


def new_plot(figsize=(20,6), fontsize=18) -> plt.subplots:
    '''Return a created new plotframe.'''
    
    fig,  ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Time (hr)', fontsize=fontsize)
    ax.set_ylabel('Discharge (cfs)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(which='major', color='lightgrey', linestyle='--', linewidth=2)
    return fig, ax


def plot_scatter(x:np.array, y:np.array, figsize=(20,6), fontsize=18) -> plt.subplots:
    '''Return a scatter plot for a given x and y values for discharge time series.'''
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x=x, y=y)
    ax.grid(which='major', color='lightgrey', linestyle='--', linewidth=2)
    ax.set_xlabel('Date/Time', fontsize=fontsize)
    ax.set_ylabel('Discharge (cfs)', fontsize=fontsize)
        
    
def plot_selected_events(events:dict, dfslice:pd.DataFrame, int_increment:float, plot_type:str, legend=True) -> plt.subplots:
    '''Return a plot with the selected events hydrographs from the events dictionary. 
    This is to show the overall hydrograph pattern of the selected hydrograph.
    
    There are two plot types:
        1) real: plot the real retrieved data without any smoothing.
        2) interpolated: plot a smoothed retrieved data.'''
    
    fig,  ax =  new_plot()
    
    for k, v in events.items():
        tmp = dfslice.loc[v].copy()
        if tmp.shape[0] > 5:
            tmp, cum_hr_max, f = tmp_df(tmp, v)
            
            if plot_type == "interpolated":
                x = np.append(np.arange(0,cum_hr_max, cum_hr_max/int_increment),cum_hr_max)
                ax.plot(x, 
                        f(x), 
                        label=str(tmp.loc[tmp['flow'] == tmp['flow'].max(), 'dateTime'].values[0]).split('T')[0])
            
            if plot_type == "real":
                ax.plot(tmp['cumulative_hours'], tmp['flow'], label=str(tmp['dateTime'].values[0]).split('T')[0])
            
            if legend:
                ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 18})
                

def plot_final_hydrographs(bin_return_pd:dict, mean_hydrograph_x:dict, mean_hydrograph_y:dict, legend=True) -> plt.subplots:
    '''Return a plot for the final generated hydrograph for a selected return period.'''
    
    for bin, yr in bin_return_pd.items():
        if len(yr) > 0.0:
            fig, ax = new_plot()
            for i in yr:
                ax.plot(mean_hydrograph_x[i], mean_hydrograph_y[i], label=str(round(i,1)) + " Yr")
            if legend:
                ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 12})


def plot_hydrographs(bin_sv:float, bin_ev:float, int_increment:float, dfslice:pd.DataFrame, bins_avg_cum_hr:dict, events:dict, excluded_dates:list, plot_type:str, data_threshold:int, legend=True, generate_m_h=False) -> plt.subplots:
    '''Return a separate plot for each bin.
    
    There are three plot types:
        1) real: plot the real retrieved data without any smoothing.
        2) interpolated: plot a smoothed retrieved data.
        3) interpolated_normalized: plot with smoothed and normalized retieved data.'''
    
    fig,  ax =  new_plot()

    for k, v in events.items():
            tmp = dfslice.loc[v].copy()
            if tmp.shape[0] > 5:
                tmp, cum_hr_max, f = tmp_df(tmp, v)
                x_f = np.append(np.arange(0,cum_hr_max, cum_hr_max/int_increment),cum_hr_max)
                y_f = f(x_f)
                y_f_min_max = (y_f - y_f.min())/(y_f.max() - y_f.min())                

                if str(tmp.loc[tmp['flow'] == tmp['flow'].max(), 'dateTime'].values[0]).split('T')[0] not in excluded_dates:                    

                        if bin_sv < tmp.cumulative_hours.max() <= bin_ev:
                            if plot_type == "interpolated":
                                ax.plot(x_f, y_f, label=str(tmp.loc[tmp['flow'] == tmp['flow'].max(), 'dateTime'].values[0]).split('T')[0])

                            if plot_type == "interpolated_normalized":                              
                                ax.plot(x_f, y_f_min_max, label=str(tmp.loc[tmp['flow'] == tmp['flow'].max(), 'dateTime'].values[0]).split('T')[0])
                            if plot_type == "real":
                                ax.plot(tmp['cumulative_hours'], tmp['flow'], label=str(tmp.loc[tmp['flow'] == tmp['flow'].max(), 'dateTime'].values[0]).split('T')[0])
                            
                            fig_title = "Bins for  durations between {} and {} (hr)".format(bin_sv, bin_ev) 
    
    if generate_m_h:
        avg_h_x, avg_h_y, flow_df_normailzed  = hydrograph_generation(bin_sv, bin_ev, int_increment, dfslice, bins_avg_cum_hr, events, excluded_dates, data_threshold)
        ax.plot(avg_h_x, avg_h_y, "black", linewidth=4.0, label = 'Average Hydrograph')    
    
    ax.set_title(fig_title, fontsize=20)

    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 18})        
    
        
def hydrograph_group_plot(selected_bins:list, dfslice:pd.DataFrame, events:dict, bins_avg_cum_hr:dict, int_increment:float, plot_type:str, data_threshold: int, excluded_dates: list=None, legend=True, generate_m_h=False) -> plt.subplots:
    '''Return a group plots based on the selected bins.
    
    There are three plot types:
        1) real: plot the real retrieved data without any smoothing.
        2) interpolated: plot a smoothed retrieved data.
        3) interpolated_normalized: plot with smoothed and normalized retieved data.'''
    if excluded_dates==None:
        excluded_dates = []
    for i in range(len(selected_bins)-1):
        plot_hydrographs(selected_bins[i], selected_bins[i+1], int_increment, dfslice, bins_avg_cum_hr, events, excluded_dates, plot_type, data_threshold, legend, generate_m_h)


def normhydro_dic(dfslice: pd.DataFrame, events: dict, selected_peaks: dict, 
                  bins_avg_cum_hr: dict, selected_bins: list, 
                  excluded_dates: list, data_threshold: int, 
                  int_increment: int) -> dict:
    """Construct a dictionary to store the unit hydrographs and associated
       metadata.
    """
    df = dfslice.copy(deep=True)
    df['value'] = [int(x) for x in list(df['value'])]
    df['dateTime'] = [datetime.strftime(x, '%Y-%m-%d %H:%M:%S') 
                      for x in list(df['dateTime'])]
    df['flow'] = [int(x) for x in list(df['flow'])]
    df['increment_hr'] = [x.total_seconds() for x in list(df['increment_hr'])]
    events_int = {k:[int(x) for x in v] for (k, v) in events.items()}
    normhydro = {'data_threshold': data_threshold, 
                 'selected_bins': selected_bins, 
                 'selected_peaks': selected_peaks, 
                 'dfslice': df.to_dict(), 
                 'bins_avg_cum_hr': bins_avg_cum_hr, 
                 'events': events_int, 
                 'int_increment': int_increment, 
                 'excluded_dates': excluded_dates}
    return normhydro


register_matplotlib_converters()