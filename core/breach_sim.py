'''
Development Script for probrisk module

slawler@dewberry.com
9.21.2018
'''
import os
import random
import collections
import pprint
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
pd.set_option('precision',16)

# Calculate Breach Probability 
def BreachProb(wse, levee_elevations, failure_probs):
    '''Calculate Breach Probability given curve data & wse'''
    
    # If WSE is below the Levee Toe, Probability = 0
    if wse < min(levee_elevations):
        return 0
    
    # If WSE is between the Levee Toe & Crest, Interpolate using fragility curve
    elif wse < max(levee_elevations):
        return np.interp(wse, levee_elevations, failure_probs)
    
    # If WSE is at or above the Levee Crest, Probability = 1    
    else:
        return 1

# Plot Fragility Curves
def plot_performance(df, breach_location):
    assert isinstance(df, pd.core.frame.DataFrame) 
    fig, ax =plt.subplots(figsize=(16,2))

    ax.plot(df.loc['{} System Response'.format(breach_location)][0:-1]*100,
            df.loc['{} River Elevation'.format(breach_location)][0:-1])

    ax.set_xlabel('Chance of Failure (%)')
    ax.set_ylabel('Elevation (ft) \nLevee Toe to Crest')
    ax.set_title('Performance Graph \n {}'.format(breach_location))
    ax.grid()

# Read in wse data & fragility curves, output raw prob table
def compute_raw_prob(wses, df_frag, raw_prob_table):
    assert isinstance(wses, pd.core.frame.DataFrame) 
    assert isinstance(df_frag, pd.core.frame.DataFrame) 
    assert isinstance(raw_prob_table, str) 

    df_raw_probs = pd.DataFrame(index=wses.index)

    for col in wses.columns:
        wse_data = wses[[col]]
        curve_data = df_frag[df_frag.index.str.contains(col, regex=False)]
        df_raw_probs['RawProb_' + col] = [float(BreachProb(wse, curve_data[:].iloc[0].values, curve_data[:].iloc[1].values)) for wse in wse_data.values]
        df_raw_probs.to_csv(raw_prob_table, sep='\t')  
    return df_raw_probs


# Read Input Table 
def readtable(table, prefix='RawProb'):
    '''prefix= String part indicating columns to include in calculations.  Run must be first column'''
    run2data = collections.OrderedDict()
    for idx, line in enumerate(open(table)):
        parts = line.strip().split()
        if idx == 0:
            titles = parts[1:]
            nbreaches =  len(tuple(title for title in titles if title.startswith(prefix)))   
        else:
            assert len(parts[1:]) == len(titles)
            run = parts[0]
            data = {}
            for title, chancetxt in zip(titles, parts[1:]):
                chance = float(chancetxt)
                data[title] = chance
            assert run not in run2data
            run2data[run] = data

    return run2data, nbreaches

# Monte Carlo Simulation
def simulate_mb(rundata, nbreaches, multi_breach =True, prefix ='B', RANDCOUNT = 100000):
    assert RANDCOUNT%100000==0
    assert RANDCOUNT <= 1e6

    '''MC approach for ranodm sampling of breaches'''
    breachtitles = tuple(key for key in rundata if prefix in key)
    short_breach_titles = [title.split('_')[1] for title in breachtitles]
    #breachtitles = tuple(key for key in rundata if key.startswith("rawprobBL-"))
    
    leveecount = len(short_breach_titles)
    assert leveecount == nbreaches 

    randomarr = np.random.random((RANDCOUNT, nbreaches))
    cutoffchances = np.array(tuple(rundata[title] for title in breachtitles), dtype="d")
    
    nobreach_chance = 1
    for chance in cutoffchances:
        nobreach_chance *= (1 - chance)
    assert len(cutoffchances) == nbreaches
    #print("Theoretical breach chance anywhere {}".format(1 - nobreach_chance))
    
    short_title2breachcount = collections.defaultdict(int)
    
    for row in range(RANDCOUNT):
        randnums = randomarr[row]
        assert len(randnums) == nbreaches
        
        rawbreachbool = (cutoffchances > randnums)
        breachcount = np.sum(rawbreachbool)
        
        if breachcount == 0:
            continue
            
        elif breachcount == 1:
            argmax = np.argmax(rawbreachbool)
            title = short_breach_titles[argmax]
            short_title2breachcount[title] += 1
            
        else:
            assert nbreaches >= breachcount > 1
            
            breachidxs = rawbreachbool.nonzero()[0]
            assert len(breachidxs) == breachcount

            if multi_breach:
                title = ','.join(short_breach_titles[i] for i in breachidxs) #e.g. "bl-1,bl-4"
                #print(title)
            else:    
                randidx = random.choice(breachidxs)
                assert randidx in breachidxs
            
                title = short_breach_titles[randidx]
                
            short_title2breachcount[title] += 1
    
    short_title2chance = collections.OrderedDict()
    
    for title in short_title2breachcount:
        assert not title.startswith(prefix)
        breachcount = short_title2breachcount[title]
        chance = breachcount / RANDCOUNT
        short_title2chance[title] = chance
        
    short_title2chance['_nobreach'] = 1-sum(short_title2chance.values())
    return short_title2chance


def run_sim(run2data, nbreaches, sim_results_table, multi_breach=True,  
            prefix = 'RawProb', print_stdout=False):
    assert isinstance(run2data, collections.OrderedDict) 
    #assert os.path.exists(sim_results_table) 
    
    start = datetime.now()
    titles = None
    fout = open(sim_results_table, "w")
    fout.write('Event\tChance\tBreachCombo\n')

    for event in sorted(run2data, reverse=False):
        data = run2data[event]
        sim_results = simulate_mb(data, nbreaches, multi_breach, prefix)

        for breach_combo in sorted(sim_results, key=breach_sorter):
            
            if sim_results[breach_combo]:
                datarow = '{}\t{:8.15f}\t{}'.format(event, sim_results[breach_combo], breach_combo) 
                
                fout.write(datarow + "\n")
                if print_stdout:
                    print(datarow)


        fout.flush()
    fout.close()
    print('\nTotal Compute time: {}'.format(datetime.now() - start))
    df = pd.read_csv(sim_results_table, sep='\t')
    print('Results written to {}'.format(sim_results_table))
    
    return df

#  Function written for debugging 
def breach_sorter(breach_name):
    if ',' in breach_name:
        return breach_name
    else:
        return '_' + breach_name


#-----------------------------------------------------------------------------#
# Helper functions --> not required to compute results
def get_required_sims(df):
    assert isinstance(df, pd.core.frame.DataFrame) 
    required_sims = []
    for idx in df.index:
        event = df.loc[idx, 'Event']
        breach_runs = df.loc[idx, 'BreachCombo'].split(',')
        for b in breach_runs:
            if '_nobreach' not in b:
                required_sims.append(str(event)+','+str(b))
                
    required_sims = set(required_sims)        
    print('Total # of Breach Simulations required: {}'.format(len(required_sims)))
    return set(required_sims)
    
def get_event_info(df, event):
    assert isinstance(df, pd.core.frame.DataFrame) 
    table = pd.pivot_table(df,  columns = ['BreachCombo'], index=['Event'])
    table = table['Chance']
    return pd.DataFrame(table.loc[event].dropna())

def single_breach_event(df, event):
    assert isinstance(df, pd.core.frame.DataFrame) 
    print('\nSingle Breach Probalities computed for event {}\n\t'.format(event))
    return get_event_info(df, event)

def multi_breach_event(df, event):
    assert isinstance(df, pd.core.frame.DataFrame) 
    print('\nMulti Breach Probalities computed for event {}\n\t'.format(event))
    return get_event_info(df, event)



# Reporting Notebooks


def plot_wse_at_levee(breach_location, levee_crest, levee_toe, events_axis, event_wses):
    fig, ax =plt.subplots(figsize=(16,6))
    # WSE Results
    ax.scatter(x = events_axis, y=event_wses, marker = '*', color = 'black', label = 'Peak Stage')
    
    # Overtopping Region
    max_overtop = np.max([levee_crest, event_wses.max()])
    ax.fill_between(np.arange(0, events_axis.max()+2) , 
                    levee_crest,max_overtop, color='orange', alpha = 0.3, label='Overtopping')
    
    # Levee Crest
    ax.hlines(levee_crest, 0,events_axis.max()+2, label='Levee Crest', color = 'red')
    
    # Levee
    ax.fill_between(np.arange(0,events_axis.max()+2) ,levee_toe, 
                    levee_crest, color='green', alpha = 0.3, label='Levee')

    # Levee Toe
    ax.hlines(levee_toe, 0,events_axis.max()+2, label='Levee Toe', color = "black")

    # Channel
    ax.fill_between(np.arange(0,events_axis.max()+2) ,event_wses.min(), 
                    levee_toe, color='blue', alpha = 0.3, label='Channel')

    # Plot Text
    ax.set_xlabel('Modeled Events')
    ax.set_ylabel('Elevation (ft)')
    ax.set_xlim(0, events_axis.max()+1)
    ax.set_title("Riverside WSE's \n No-Breach Scenario at {}".format(breach_location))
    ax.legend()
    ax.grid()







    