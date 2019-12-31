from collections import namedtuple, defaultdict, OrderedDict
import gdal
import pandas as pd
import geopandas as gpd
import numpy as np
import boto3
from io import BytesIO
import zipfile
from shapely.geometry import Point
from shapely.wkb import loads
from scipy.interpolate import interp1d
import os
from os.path import join, exists
from matplotlib import pyplot as plt


# Define Global Variables
WSE = 'WSE'
AAL = 'AAL'
FAAL = 'FinalAAL'
SUM = 'Summary'
ANOM = 'Anomaly'
WEIGHTS = 'Weights'

curve_groups = {'singFam_1Story_NoBasement' : [105, 129, 132, 139, 173],
             'singFam_2Story_NoBasement' : [107, 130, 136, 140, 174],
             'singFam_3Story_NoBasement' : [109],
             'singFam_1Story_Basement' : [106, 121, 133, 181, 704],
             'singFam_2Story_Basement' : [108, 122, 137],
             'singFam_3Story_Basement' : [110, 123],
             'mobileHome': [189, 191, 192, 203]}

class PointRisk:
    def __init__(self, plus_code, risk_data):
        self._risk_data = risk_data
        self.__plus_code = plus_code
        self.__risk_fields = [c for c in list(self._risk_data.index.values)]
        self.__wses_fields = [c for c in self.__risk_fields if '_E' in c]
        self.__attr_fields = [c for c in self.__risk_fields if not '_E' in c and not 'geom' in c]
        self._pfra_elevation = self._risk_data['GroundElev' ]
        self._simulations = set([c.split('_')[0] for c in self.events])
        
        try:
            self.__point = loads(self._risk_data.geom,hex=True)
        except:
            # Create from lat/lon
            pass
    
    @property    
    def plus_code(self):
        return self.__plus_code
    
    @property    
    def point(self):
        return self.__point
    
    @property  
    def events(self):
        return self.__wses_fields
        
    @property
    def simulations(self):
        return self._simulations
    
    @property
    def attr_fields(self):
        return self.__attr_fields
    
    @property
    def attributes(self):
        attrs = [c for c in self.__risk_fields if '_E' not in c and 'geom' not in c]
        return self._risk_data[attrs]
    
    @property
    def elevation(self):
        assert 'GroundElev' in self.__risk_fields, 'No GroundElev found, please check risk_data'
        return self._risk_data['GroundElev' ]
    
    @property
    def raw_wses(self):
        """Raw output from post processing"""
        wses_fields = [c for c in self.__risk_fields if '_E' in c]
        return self._risk_data[wses_fields]
    
    @property
    def abs_wses(self):
        """Nans replaced with GroundElevation for analysis"""
        return self._risk_data[self.__wses_fields].fillna(self._pfra_elevation)
    
    @property
    def depths(self):
        return self._risk_data[self.__wses_fields].fillna(self._pfra_elevation) - self._pfra_elevation
         
        

class PluvialPoint(PointRisk):       
    
    def __init__(self, plus_code, risk_data, pfra_type='Pluvial'):
        
        super().__init__(plus_code, risk_data)
        
        self._pfra_type = pfra_type
        
        
        
class FluvialPoint(PointRisk):  
    
    def __init__(self, plus_code, risk_data, pfra_type='Fluvial'):
        
        super().__init__(plus_code, risk_data)
        
        self._pfra_type = pfra_type  
        self.__nbr_fields = [c for c in self.events if 'NBR' in c]
        self.__breach_fields = [c for c in self.events if 'NBR' not in c]


    @property
    def breach_locations(self):
        return set(x for x in self._simulations if 'NBR' not in x)
    
    @property
    def nbr_events(self):
        return self.__nbr_fields
    
    @property
    def raw_nbr_results(self):
        return self._risk_data[self.__nbr_fields]
    
    @property
    def abs_nbr_results(self):
        return self.abs_wses[self.__nbr_fields]
    
    def abs_breach_results(self, breach_location):
        return self.abs_wses[[b for b in self.__breach_fields if breach_location in b]]
    
    @property
    def breach_influence(self, epsilon = 1e-1):
        use_breaches=[]
        nt = namedtuple('BreachRuns', 'event value')
        nd = namedtuple('NBR', 'event value')

        b = self.abs_nbr_results
        idxs = [d.split('_')[1] for d in b.index]
        nresults = dict(map(nd, idxs, b.values))

        for breach_location in self.breach_locations: 
            b = self.abs_breach_results(breach_location)
            idxs = [d.split('_')[1] for d in b.index]
            bresults = dict(map(nt, idxs, b.values))

            for k,v in nresults.items():
                if k in bresults.keys():
                    if abs(bresults[k] - v) > epsilon:
                        use_breaches.append(breach_location)
                        break
        return use_breaches


    
class FluvialEvents:
    
    def __init__(self, df_weights, df_breach_prob):
        self._event_ids = df_weights.index
        self._event_weights = df_weights
        self._breach_probs = df_breach_prob
        self._breach_locations =  self._breach_probs.columns.to_list()
        self._breach_events =  self._breach_probs.index.to_list()
        
        assert all(self._breach_events == self._event_ids), 'Events in Breach Prob file do not match events in weights file'
        
    @property
    def breach_locations(self):
        return self._breach_locations
    
    @property
    def event_weights(self):
        assert abs(0.50 - self._event_weights.sum().sum()) < 1e6, 'Event weights do not sum to 0.50, please check inputs'
        return self._event_weights

    def breach_probs(self, breach_location):
        return self._breach_probs[breach_location]
    
    def event_weights_for_point(self):
        pass
        
def setup_file_structure(out_dir, mods):
    """Set up the directories for output files, return json of directories"""
    dir_json = {}
    for m in mods:
        dir_json[m] = {}
        weight_dir = join(out_dir, m, WEIGHTS)
        if not exists(weight_dir): os.makedirs(weight_dir)
        dir_json[m][WEIGHTS] = weight_dir

        wse_dir = join(out_dir, m, WSE)
        if not exists(wse_dir): os.makedirs(wse_dir)
        dir_json[m][WSE] = wse_dir

        aal_dir = join(out_dir, m, AAL)
        if not exists(aal_dir): os.makedirs(aal_dir)
        dir_json[m][ AAL] = aal_dir

    final_aal_dir = join(out_dir, FAAL)
    if not exists(final_aal_dir): os.makedirs(final_aal_dir)
    dir_json[FAAL] = final_aal_dir

    summary_dir = join(out_dir, SUM)
    if not exists(summary_dir): os.makedirs(summary_dir)
    dir_json[SUM] = summary_dir

    anomaly_dir = join(out_dir, ANOM)
    if not exists(anomaly_dir): os.makedirs(anomaly_dir)
    dir_json[ANOM] = anomaly_dir

    return dir_json


def calc_pluv_aal_mp(args):
    """Pure python version of the AAL calculator, built for multiprocessing Pool"""
    # unpack args
    pcode, wse_dict, weights_dict, col_events, loss_functs, lowest_ddf_elev, highest_ddf_elev, structure_cols = args
    
    bld_lmt = wse_dict[pcode][structure_cols['Building Limit']]
    bld_ded = wse_dict[pcode][structure_cols['Building Deduction']]
    dmg_code = wse_dict[pcode][structure_cols['Damage Code']]
    elev = wse_dict[pcode][structure_cols['Ground Elevation']]
    ffh = wse_dict[pcode][structure_cols['First Floor Height']]
    
    aal = 0
    for event in col_events:
        wse = wse_dict[pcode][event]
        depth = wse - (elev + ffh)
        if not depth < lowest_ddf_elev:
            if depth > highest_ddf_elev:
                depth = highest_ddf_elev
            percent_loss = loss_functs[dmg_code](depth)/100
            loss_val = bld_lmt * percent_loss - bld_ded
            if loss_val > 0:
                aal += loss_val * weights_dict[event]
    return (pcode, aal)


def hazusID_to_depth(df: any) -> any:
    """Formats Hazus Depth In Structure vs Damages table to be readable 
    for this script.
    """
    rawCol, rawDIS, newDIS = df.columns.tolist(), [], [] 
    for col in rawCol:
        if col[0] == 'm' or col[0] == 'p':
            rawDIS.append(col)
            newcol = int(col.replace('m','-').replace('p',''))
            newDIS.append(newcol)
    for i, col in enumerate(rawDIS):
        df.rename(columns={col:newDIS[i]}, inplace=True)
    return df

def aggregate_ddf_curves(df: any, curve_groups: dict, 
                         custom_depths: list = False, 
                         plot: bool = True) -> any:
    '''curve_groups is a dictionary categrizing a list of damage 
    functions to aggregate e.g. "Category1": [1,2,3].
    '''
    depths_in_curves = custom_depths if custom_depths else list(range(-4,25))
    df_agg = pd.DataFrame()
    for group in curve_groups.keys():
        dfc = df.loc[curve_groups[group]][depths_in_curves].T
        occ_type =  df['Occupancy'].loc[curve_groups[group]].unique()[0]
        df_agg[group] = dfc.mean(axis=1)
        if plot:
            fig, ax = plt.subplots(figsize=(22,4))
            for idx in dfc.columns:
                ax.plot(dfc[idx], linestyle='--', label =str(idx))
            ax.plot(dfc.mean(axis=1), label='Mean', color='black')
            ax.set_title(f'Raw Depth Damage curves for {group}' \
                         + f'\n({occ_type})',fontsize=20)
            ax.legend()
            ax.grid()
            ax.set_xlabel('Depth (ft)', fontsize=16)
    return df_agg


def calculate_breach_weights(event:str ,breach_prob:dict) -> dict:

    """
    the input dictionary breach_prob should contain one key for each breach label.
    Each breach label should be assigned a probability of breach from 0 to 1
    input example:
    test_breach_prob = {'B1':0.00,
                        'B2':0.48,
                        'B3':0.15,
                        'B4':0.04}
    outputs is dictionary with weight assigned to each breach, plus a no_breach scenario
    outputs example:
                        {'no_breach': 0.42432,
                         'B1': 0.0,
                         'B2': 0.4124274626865671,
                         'B3': 0.12888358208955222,
                         'B4': 0.034368955223880594}
    The output weights sum to 1.
    The weights are used to calculate average flooding
    """

    #get probability of breach and no breach
    #also get sum of all breach probabilities
    #first initialize variables
    p_no_breach = 1
    sum_breach_prob = 0
    #loop through each breach id in the dictionary
    for breach_id in breach_prob.keys():
        p_no_breach = p_no_breach * (1-breach_prob[breach_id])
        sum_breach_prob = sum_breach_prob + breach_prob[breach_id]
    #get probability of breach
    p_breach = 1 - p_no_breach

    #create new dictionary with weight for each breach
    breach_weight = {}
    if sum_breach_prob==0:
        breach_weight['NBR_{}'.format(event)] = 1.0
        #assign the remaining probability to the single-breach scenarios, weighted by their probability
        for breach_id in breach_prob.keys():
            breach_weight['{}_{}'.format(breach_id, event)] = 0.0

    else:
        #assign the no_breach scenario with a weight of the no-breach-probability
        breach_weight['NBR_{}'.format(event)] = p_no_breach

        #assign the remaining probability to the single-breach scenarios, weighted by their probability
        for breach_id in breach_prob.keys():
            breach_weight['{}_{}'.format(breach_id, event)] = p_breach * (breach_prob[breach_id] / sum_breach_prob)

    #error detection: check that the final weights sum to 1 +/- 0.01
    #if not return error
    sum_breach_weight = 0
    for breach_id in breach_weight.keys():
        sum_breach_weight = sum_breach_weight + breach_weight[breach_id]
    assert abs(sum_breach_weight-1.0) < 0.01, ('Error: breach weights do not sum to 1.0')

    return breach_weight
        

def map_relative_weights(breach_prob, breaches, e_event_weights, p):
    """Given breaches that impact a point, event weights and breach probabilities,
       create dictionary mapping relative distribution of weights for a point 
       for each event 
    """
    
    relative_weights = defaultdict(dict, {event:{} for event in e_event_weights['Overall Weight'].keys()})

    if len(breaches) == 0:
        for event in relative_weights.keys():
            relative_weights[event]['NBR_{}'.format(event)] = 1

    else:
        for event in relative_weights.keys():
            events_mapper = defaultdict(dict)
            for breach_id in breaches:
                sim_id = f'{breach_id}_{event}'
                if sim_id in p.abs_breach_results(breach_id).keys():
                    events_mapper[event][breach_id] = breach_prob[breach_id][event]

            relative_weights[event] = calculate_breach_weights(event, events_mapper[event])

    return relative_weights


def map_simulation_weights(relative_weights, e_event_weights):
    weights_map = {}
    for event in relative_weights.keys():
        for sim_id, relative_wt in relative_weights[event].items():
            absolute_wt = relative_wt * e_event_weights['Overall Weight'][event]
            weights_map[sim_id] = absolute_wt 
    return weights_map


def loss_calculation(sim_weights, depth_in_building, min_value, max_value, loss_func, p, structure_cols):
    bld_lmt  = p.attributes[structure_cols['Building Limit']]
    bld_ded  = p.attributes[structure_cols['Building Deduction']]
    dmg_code = p.attributes[structure_cols['Damage Code']]
    ffh      = p.attributes[structure_cols['First Floor Height']]

    average_annual_loss = 0
    for event in sim_weights.keys():
        try:
            event_depth_in_building = depth_in_building[event]
        except KeyError:
            #print(f'Unable to compute {event}')
            event_depth_in_building=0

        if event_depth_in_building > min_value:

            if event_depth_in_building > max_value:
                event_depth_in_building = max_value

            percent_loss = loss_func(event_depth_in_building)/100
            loss_val = bld_lmt * percent_loss - bld_ded

            if loss_val > 0:
                try:
                    average_annual_loss += loss_val * sim_weights[event]
                except KeyError as error:
                    print('Key Error', error)
                    break
                    
    return average_annual_loss


def calc_fluv_aal_mp_functions(allarguments):
    """Main function to calculate AAL at a point"""
    pcode, global_model_data, df_agg, sim_results, structure_cols = allarguments
    
    
    p = FluvialPoint(pcode, sim_results)
    dmg_code = p.attributes[structure_cols['Damage Code']]
    breaches  = sorted(p.breach_influence)
    
    e_event_weights = global_model_data.event_weights.to_dict()
    raw_breach_probs = {b : global_model_data.breach_probs(b).to_dict() for b in breaches}
    
    relative_weights =  map_relative_weights(raw_breach_probs, breaches, e_event_weights, p)
    
    sim_weights = map_simulation_weights(relative_weights, e_event_weights)
    loss_func = interp1d(df_agg.index, df_agg[dmg_code])
    max_value = df_agg.index.max()
    min_value = df_agg.index.min()
    
    depth_in_building = p.raw_wses - (p.elevation + p.attributes[structure_cols['First Floor Height']])
    aal = loss_calculation(sim_weights, depth_in_building, min_value, max_value, loss_func, p, structure_cols)
    return (pcode, aal)
