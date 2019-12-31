import pandas as pd
import numpy as np
from datetime import datetime
import os
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from scipy.optimize import minimize
from scipy import interpolate
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.offline as po
init_notebook_mode()
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.offline as po



'''
Plotting functions for Mean Curve Notebooks
'''



def plot_ssp_curves(df):
    fig, ax = plt.subplots(figsize = (24,10))
    for col in df.columns:
        if col == 50.0:
            ax.semilogx(df.index, df[col], label='Median Value'.format(col), color='black', linewidth=4)
        else:
            ax.semilogx(df.index, df[col], label='{} CL'.format(col))
        ax.set_title('Selected Confidence Limits\n(Generated using HEC-SSP)')
        ax.set_xlabel('Return Period')
        ax.set_ylabel('Peak Flow (CFS)')

    #ax.scatter(rand_flow_data.index/100,rand_flow_data.values )        
    fig.gca().invert_xaxis()    
    ax.legend();ax.grid()


def plot_ssp_curves(df):
    
    fig, ax = plt.subplots(figsize = (24,10))
    for col in df.columns:
        if col == 50.0:
            ax.semilogx(df.index, df[col], label='Median Value'.format(col), color='black', linewidth=4)
        else:
            ax.semilogx(df.index, df[col], label='{} CL'.format(col))
        ax.set_title('Selected Confidence Limits\n(Generated using HEC-SSP)')
        ax.set_xlabel('Return Period')
        ax.set_ylabel('Peak Flow (CFS)')

    #ax.scatter(rand_flow_data.index/100,rand_flow_data.values )        
    fig.gca().invert_xaxis()    
    ax.legend();ax.grid()


#Functions added by sputnam:
def plot_ssp_transform(data,AEPz,CLz):
    cmap = mpl.cm.viridis    
    cmap1 = mpl.cm.viridis_r
    fig=plt.figure(1, figsize=(24,6))
    plt.clf()
    ax1=plt.subplot2grid((1,2),(0,0), xlabel='Annual Exceedance Probability, [z variate]', ylabel='Discharge, [log(cfs)]', title='Discharge vs. Annual Exceedance Probability (Each Line is a Confidence Limit)')
    for i in np.arange(len(CLz)):
        ax1.plot(AEPz,data.iloc[:,i], linestyle="-",marker='.', color=cmap(i / len(CLz)))
    #ax1.set_xlim([-2.5,5.5])
    #ax1.set_ylim([2.5,6.5])
    ax1=plt.subplot2grid((1,2),(0,1), xlabel='Confidence Limits, [z variate]', ylabel='Discharge, [log(cfs)]', title='Discharge vs. Confidence Limits (Each Line is an Annual Exceedance Probability)')
    for i in np.arange(len(AEPz)):
        ax1.plot(CLz,data.iloc[i], linestyle="-",marker='.', color=cmap1(i / len(AEPz)))
    #ax1.set_xlim([-2.5,2.5])
    #ax1.set_ylim([2.5,6.5])

def plot_ssp_interp(df2,CLz):
    cmap = mpl.cm.viridis    
    fig=plt.figure(2, figsize=(12,6))
    plt.clf()
    ax1=plt.subplot2grid((1,1),(0,0), xlabel='Confidence Limits, [z variate]', ylabel='Annual Exceedance Probability, [z variate]', title='Annual Exceedance Probability vs. Confidence Limits (Each Line is a Discharge)')
    for i in np.arange(len(df2.iloc[:])):
        ax1.plot(CLz,df2.iloc[i], linestyle="-",marker='.', color=cmap(i / len(df2.iloc[:])))
    #ax1.set_xlim([-2.5,2.5])
    #ax1.set_ylim([-5,20])

def plot_ssp_interptrans(df3):
    cmap = mpl.cm.viridis    
    fig=plt.figure(3, figsize=(12,6))
    plt.clf()
    ax1=plt.subplot2grid((1,1),(0,0), xlabel='Confidence Limits', ylabel='Annual Exceedance Probability', title='Annual Exceedance Probability vs. Confidence Limits (Each Line is a Discharge)')
    for i in np.arange(len(df3.index)):
        ax1.plot(df3.columns.values,df3.iloc[i], linestyle="-",marker='.', color=cmap(i / len(df3.index)))
    #ax1.set_xlim([0,1])
    #ax1.set_ylim([0,1.2])

def plot_ssp_meanmed(AEPz,df1,AEPmz,Q):
    fig=plt.figure(4, figsize=(12,6))
    plt.clf()
    ax1=plt.subplot2grid((1,1),(0,0), xlabel='Annual Exceedance Probability, [z variate]', ylabel='Discharge, [log(cfs)]', title='Discharge vs. Annual Exceedance Probability, with Mean and Median Curves')
    ax1.plot(AEPz,df1.iloc[:]['0.5'], linestyle="-",marker='.', label='Median', color='black')
    ax1.plot(AEPmz,Q, linestyle="-",marker='.', label='Mean', color='red')
    #ax1.set_xlim([-2.5,7.5])
    #ax1.set_ylim([2.5,6.5])
    ax1.legend(loc='lower right',frameon=False)
    
def plot_ssp_meanmedffc(table, gage_ID, iplot=False):
    fig, ax=plt.subplots(1,2, figsize=(24,6))
    ax[0].plot([i*100 for i in table.index],table['Q_Median_cfs'], linestyle="-",marker='.', label='Median', color='black')
    ax[0].plot([i*100 for i in table.index],table['Q_Mean_cfs'], linestyle="-",marker='.', label='Mean', color='red')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Annual Exceedance Probability, [%]')
    ax[0].set_ylabel('Discharge, [cfs]')
    ax[0].set_title('Discharge vs. Annual Exceedance Probability (Station ID:%s)'%gage_ID)
    ax[0].grid(True, which="both") 
    ax[0].legend(loc='lower left',frameon=True) 
    ax[0].set_xticklabels(['{:}'.format(x) for x in ax[0].get_xticks()])

    ax[1].plot([1.0/i for i in table.index],table['Q_Median_cfs'], linestyle="-",marker='.', label='Median', color='black')
    ax[1].plot([1.0/i for i in table.index],table['Q_Mean_cfs'], linestyle="-",marker='.', label='Mean', color='red')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Recurrence Interval')
    ax[1].set_ylabel('Discharge, [cfs]')
    ax[1].set_title('Discharge vs. Recurrence Interval (Station ID:%s)'%gage_ID)
    ax[1].grid(True, which="both") 
    ax[1].legend(loc='lower right',frameon=True)
    if iplot:
        plt.close(fig)
    return fig

def plot_ssp_meanmedffc_events(table, df3, gage_ID):
    fig, ax=plt.subplots(1,2, figsize=(24,6))
    ax[0].plot([i*100 for i in table.index],table['Q_Median_cfs'], linestyle="-",marker='', label='Median', markersize=12, color='black')
    ax[0].plot([i*100 for i in table.index],table['Q_Mean_cfs'], linestyle="-",marker='', markersize=12, label='Mean', color='red')
    for i in range(0,len(df3.columns)-1):
    	ax[0].plot(df3.AEP,df3.iloc[:, [i]], linestyle="",marker='.', label='', color='blue')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Annual Exceedance Probability, [%]')
    ax[0].set_ylabel('Discharge, [cfs]')
    ax[0].set_title('Discharge vs. Annual Exceedance Probability (Station ID:%s)'%gage_ID)
    ax[0].grid(True, which="both") 
    ax[0].legend(loc='lower left',frameon=True) 
    ax[0].set_xticklabels(['{:}'.format(x) for x in ax[0].get_xticks()])

    ax[1].plot([1.0/i for i in table.index],table['Q_Median_cfs'], linestyle="-",marker='', label='Median', color='black')
    ax[1].plot([1.0/i for i in table.index],table['Q_Mean_cfs'], linestyle="-",marker='', label='Mean', color='red')
    for i in range(0,len(df3.columns)-1):
    	ax[1].plot(df3.index,df3.iloc[:, [i]], linestyle="",marker='.', label='', color='blue')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Recurrence Interval')
    ax[1].set_ylabel('Discharge, [cfs]')
    ax[1].set_title('Discharge vs. Recurrence Interval (Station ID:%s)'%gage_ID)
    ax[1].grid(True, which="both") 
    ax[1].legend(loc='lower right',frameon=True)
    return fig



def plot_AEP_Q(df,interp):
    fig=plt.figure(1, figsize=(12,6))
    plt.clf()
    ax1=plt.subplot2grid((1,1),(0,0), xlabel='Annual Exceedance Probability, [z variate]', ylabel='Discharge, [log(cfs)]', title='Discharge vs. Annual Exceedance Probability')
    ax1.plot(interp.AEPz,interp.Qlog_int, linestyle="-",marker='.', label='Interpolated', color='blue')
    ax1.plot(df.AEPz,df.Qlog, linestyle="",marker='.', label='StreamStats', color='purple',markersize=10)
    #ax1.set_xlim([-2.5,7.5])
    #ax1.set_ylim([2.5,6.5])
    ax1.legend(loc='lower right',frameon=False)

def plot_meanffc(interp, gage_ID, iplot=False):
    fig, ax=plt.subplots(1,2, figsize=(24,6))
    ax[0].plot([i*100 for i in interp.index],interp.Q_int, linestyle="-",marker='.', label='Median', color='blue')
#    ax[0].plot([i*100 for i in interp.index],Q3, linestyle="-",marker='.', label='Mean', color='red')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Annual Exceedance Probability, [%]')
    ax[0].set_ylabel('Discharge, [cfs]')
    ax[0].set_title('Discharge vs. Annual Exceedance Probability (Station ID:%s)'%gage_ID)
    ax[0].grid(True, which="both") 
#    ax[0].legend(loc='lower left',frameon=True) 
    ax[0].set_xticklabels(['{:}'.format(x) for x in ax[0].get_xticks()])

    ax[1].plot([1.0/i for i in interp.index],interp.Q_int, linestyle="-",marker='.', label='Median', color='blue')
#    ax[1].plot([1.0/i for i in interp.index],Q3, linestyle="-",marker='.', label='Mean', color='red')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Recurrence Interval')
    ax[1].set_ylabel('Discharge, [cfs]')
    ax[1].set_title('Discharge vs. Recurrence Interval (Station ID:%s)'%gage_ID)
    ax[1].grid(True, which="both") 
#    ax[1].legend(loc='lower right',frameon=True)
#    fig.savefig(os.path.join(path,'%s.png' %gage_ID))
    if not iplot:
        plt.ioff()
    return     

def plot_mean_gageungage(table, interp, gage_ID, iplot=False):
    fig, ax=plt.subplots(1,2, figsize=(24,6))
    ax[0].plot([i*100 for i in table.index],table['Q_Median_cfs'], linestyle="-",marker='.', label='Median (Gaged)', color='black')
    ax[0].plot([i*100 for i in table.index],table['Q_Mean_cfs'], linestyle="-",marker='.', label='Mean (Gaged)', color='red')
    ax[0].plot([i*100 for i in interp.index],interp.Q_int, linestyle="-",marker='.', label='Median (Ungaged)', color='blue')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Annual Exceedance Probability, [%]')
    ax[0].set_ylabel('Discharge, [cfs]')
    ax[0].set_title('Discharge vs. Annual Exceedance Probability (Station ID:%s)'%gage_ID)
    ax[0].grid(True, which="both") 
    ax[0].legend(loc='lower left',frameon=True) 
    ax[0].set_xticklabels(['{:}'.format(x) for x in ax[0].get_xticks()])

    ax[1].plot([1.0/i for i in table.index],table['Q_Median_cfs'], linestyle="-",marker='.', label='Median (Gaged)', color='black')
    ax[1].plot([1.0/i for i in table.index],table['Q_Mean_cfs'], linestyle="-",marker='.', label='Mean (Gaged)', color='red')
    ax[1].plot([1.0/i for i in interp.index],interp.Q_int, linestyle="-",marker='.', label='Median (Ungaged)', color='blue')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Recurrence Interval')
    ax[1].set_ylabel('Discharge, [cfs]')
    ax[1].set_title('Discharge vs. Recurrence Interval (Station ID:%s)'%gage_ID)
    ax[1].grid(True, which="both") 
    ax[1].legend(loc='lower right',frameon=True)
    if not iplot:
        plt.ioff()
    return fig    

def sim_Events(runs,sorted_runs):
    fig, ax = plt.subplots(1,2, figsize=(20,6))
    fig.suptitle('Augusta, GA')
    runIDs, flows = runs.index, runs['Final Peak Flow']
    SrunIDs, Sflows = sorted_runs.index, sorted_runs['Final Peak Flow']
    ax[0].scatter(x=runIDs, y=flows, label='Peak Flow')
    ax[0].set_title('Simulated Events\n(as Sampled)')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel('Event ID')
    ax[0].set_ylabel('Flow cfs')
    ax[1].scatter(x=SrunIDs, y=Sflows, label='Peak Flow')
    ax[1].set_title('Simulated Events\n(sorted by Magnitude)')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlabel('Sorted Events')
    ax[1].set_ylabel('Flow cfs')
    return


def plotly_ssp_meanmedffc(table, gage_ID):
    x = table['AEP']*100
    y = table['Q_Mean_cfs']
    trace = go.Scatter(x = x,y = y,
                name = 'Rp', mode = 'markers',text= table.Return_Period_Year.apply(lambda x : "{0:,.0f}".format(x)) + '-year', hoverinfo='text+name')
    trace2 = go.Scatter(x = x,y = y,
                name = 'Q', mode = 'lines', text= table.Q_Mean_cfs.apply(lambda x : "{0:,.2f}".format(x)), hoverinfo='text+name',
                line = dict(
                    color = ('rgb(204, 0, 153)'))) 
    trace3 = go.Scatter(x = x,y = y,
                name = 'Event', mode = 'markers',text= table.index, hoverinfo='text+name') 
    data = [trace,trace2,trace3]
    layout = go.Layout(dict(title='Mean Discharge vs. Annual Exceedance Probability (Station ID:%s)'%gage_ID),
                   xaxis = dict(title = 'Annual Exceedance Probability, [%]', type='log', autorange=True, tickmode = 'linear'),
                   yaxis = dict(title = 'Discharge, [cfs]', type='log', autorange=True),legend= dict(orientation="h"),
                   font = dict(color = 'rgb(0,0,0)'),paper_bgcolor = 'rgb(255,255,255)',
                   plot_bgcolor = 'rgb(255,255,255)',
                   showlegend=False)
    fig = go.Figure(data=data, layout=layout)
    interactive = iplot(fig)
