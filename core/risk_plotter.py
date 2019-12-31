# Python 3.6

#---------------------------------------------------------------------------#
#-- Import Modules/Libraries -----------------------------------------------#
#---------------------------------------------------------------------------#
import os, sys, time, glob, h5py, IPython, gdal
import pandas as pd
import numpy as np
import geopandas as gpd
import pathlib as pl

from osgeo import gdal
import rasterio
from io import BytesIO
from openpyxl import load_workbook
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from IPython.display import display, Markdown, Latex
import json

#---------------------------------------------------------------------------#
#-- Type Hints for Functions -----------------------------------------------#
#---------------------------------------------------------------------------#
printextra = "Does work but only reports results through prints"
pd_df, pandas_DF = pd.DataFrame, pd.DataFrame
mpl_figure = plt.subplots
gdal_data = 'rasterband, geotransform, and gdal raster objects'


# Plot Cumulative Sum
def plot_custom_ddf_table(custom_ddf_table: pandas_DF) -> mpl_figure:
    """Plots the Depth in Structure vs Probability of Damage for a given 
    Structure Type
    """
    fig, ax = plt.subplots(figsize=(22,4))
    for col in custom_ddf_table.columns:
        ax.plot(custom_ddf_table[col], label = col)
    ax.grid()
    ax.legend()
    ax.set_ylabel('Probability of Damage', fontsize=16)
    ax.set_xlabel('WSE - FFE (ft)', fontsize=16)
    ax.set_title('Probability Damage Table \n(User Defined)', fontsize=20)
    return None

# Formatter Functions
def pct(x, pos) -> str:
    return '{:0.3f}%'.format((x*100))
def dollars(x, pos) -> str:
    return '${:,.0f}'.format((x))
def mil(x, pos) -> str:
    return '{:,}'.format(x)

def PlotDepthInStructure(idx: str,
                         df_prb: pandas_DF,
                         df_fld: pandas_DF,
                         df_AAL: pandas_DF,
                         FFH=1,
                         flood_depth_only=True) -> mpl_figure:
    """
    Plots the flood depth or depth in structure over it's probability.
    Single Structure, Single Scenario
    """
    fig, ax = plt.subplots(figsize=(20,5))
    aal = '${:,.2f}'.format(df_AAL.loc[idx,'AAL'])
    df = df_fld.copy()
    #-1 used because this is depth in structure, FFH assumed to be 1
    df[idx].fillna(-FFH,inplace=True) 
    df[idx] = df[idx] + FFH if flood_depth_only else df[idx]
    ax.semilogx(df_prb[idx], df[idx], color='#228B22')
    plt.xlim(0.5,0.00001)
    y_lowerlim = 0 if flood_depth_only else -FFH
    y_maxlim = 1 if df[idx].max() <= 1 else df[idx].max()+(df[idx].max()*0.05)
    plt.ylim(y_lowerlim,y_maxlim)
    ax.grid(which='major', )
    ax.grid(which='minor', color='#4682B4', linestyle='--') #steelblue
    xtext = 'Flood Depth' if flood_depth_only else 'Flood Depth in Structure'
    ax.set_ylabel(f'{xtext}', fontsize=16)
    ax.set_xlabel('Probability of Occurence', fontsize=16)
    ax.set_title(f'{idx}: {xtext} vs Probability\nAAL {aal}', fontsize=20)
    pct_formatter = FuncFormatter(pct)
    for axis in [ax.xaxis]:
        axis.set_major_formatter(pct_formatter)
    return None

def PlotDamagesForStructure(idx: str,
                            df_prb: pandas_DF,
                            df_lss: pandas_DF,
                            df_AAL: pandas_DF) -> mpl_figure:
    """For a Single Structure, Single Scenario"""
    fig, ax = plt.subplots(figsize=(20,5))
    aal = '${:,.2f}'.format(df_AAL.loc[idx,'AAL'])
    df = df_lss.copy()
    df[idx].fillna(0,inplace=True)
    ax.semilogx(df_prb[idx], df[idx], color='#228B22')
    plt.xlim(0.5,0.00001)
    ax.grid(which='major', )
    ax.grid(which='minor', color='#4682B4', linestyle='--') #steelblue
    ax.set_ylabel(f'Damages ($)', fontsize=16)
    ax.set_xlabel('Probability of Occurence', fontsize=16)
    ax.set_title(f'{idx}: Damages vs Probability\nAAL {aal}', fontsize=20)
    pct_formatter = FuncFormatter(pct)
    usd_formatter = FuncFormatter(dollars)
    for axis in [ax.xaxis]:
        axis.set_major_formatter(pct_formatter)
    for axis in [ax.yaxis]:
        axis.set_major_formatter(usd_formatter)
    return None

def PlotDepthInStructure_MutliScenario(idx: str,
                                       df_prbs: pandas_DF,
                                       df_flds: pandas_DF,
                                       df_AALs: pandas_DF,
                                       scenario_names: list,
                                       FFH=1,
                                       flood_depth_only=True) -> mpl_figure:
    """For a Single Structure, Multiple Scenarios"""
    fig, ax = plt.subplots(figsize=(20,5))
    maxlist = []
    for i, df_fld in enumerate(df_flds):
        df, df_prb, df_AAL = df_fld.copy(), df_prbs[i].copy(), df_AALs[i].copy()
        aal = '${:,.2f}'.format(df_AAL.loc[idx,'AAL'])
        #-1 used because this is depth in structure, FFH assumed to be 1
        df[idx].fillna(-FFH,inplace=True) 
        df[idx] = df[idx] + FFH if flood_depth_only else df[idx]
        ax.semilogx(df_prb[idx], df[idx], label = '{}: AAL {}'.format(scenario_names[i],aal))
        maxlist.append(df[idx].max())
    plt.xlim(0.5,0.00001)
    y_lowerlim = 0 if flood_depth_only else -FFH
    y_maxlim = 1 if max(maxlist) <= 1 else max(maxlist)+(max(maxlist)*0.05)
    plt.ylim(y_lowerlim,y_maxlim)
    ax.grid(which='major', )
    ax.grid(which='minor', color='#4682B4', linestyle='--') #steelblue
    ax.legend()
    xtext = 'Flood Depth' if flood_depth_only else 'Flood Depth in Structure'
    ax.set_ylabel(f'{xtext}', fontsize=16)
    ax.set_xlabel('Probability of Occurence', fontsize=16)
    ax.set_title(f'{idx}: {xtext} vs Probability', fontsize=20)
    pct_formatter = FuncFormatter(pct)
    for axis in [ax.xaxis]:
        axis.set_major_formatter(pct_formatter)
    return None

def PlotDamagesForStructure_MutliScenario(idx: str,
                                          df_prbs: pandas_DF,
                                          df_flds: pandas_DF,
                                          df_AALs: pandas_DF,
                                          scenario_names: list) -> mpl_figure:
    """For a Single Structure, Multiple Scenarios"""
    fig, ax = plt.subplots(figsize=(20,5))
    for i, df_lss in enumerate(df_lsss):
        df, df_prb, df_AAL = df_lss.copy(), df_prbs[i].copy(), df_AALs[i].copy()
        aal = '${:,.2f}'.format(df_AAL.loc[idx,'AAL'])
        df[idx].fillna(0,inplace=True)
        ax.semilogx(df_prb[idx], df[idx], label='{}: AAL {}'.format(scenario_names[i],aal))
    plt.xlim(0.5,0.00001)
    ax.grid(which='major', )
    ax.grid(which='minor', color='#4682B4', linestyle='--') #steelblue
    ax.legend()
    ax.set_ylabel(f'Damages', fontsize=16)
    ax.set_xlabel('Probability of Occurence', fontsize=16)
    ax.set_title(f'{idx}: Damages vs Probability', fontsize=20)
    pct_formatter = FuncFormatter(pct)
    usd_formatter = FuncFormatter(dollars)
    for axis in [ax.xaxis]:
        axis.set_major_formatter(pct_formatter)
    for axis in [ax.yaxis]:
        axis.set_major_formatter(usd_formatter)
    return None

def PlotDepthInStructure_MultipleIds(idxs: list,
                                     df_prb: pandas_DF,
                                     df_fld: pandas_DF,
                                     df_AAL: pandas_DF,
                                     FFH=1,
                                     flood_depth_only=True) -> mpl_figure:
    """For a Single Scenario, Multiple Structures"""
    fig, ax = plt.subplots(figsize=(20,5))
    for idx in idxs:
        aal = '${:,.2f}'.format(df_AAL.loc[idx,'AAL'])
        df = df_fld.copy()
        #-1 used because this is depth in structure, FFH assumed to be 1
        df[idx].fillna(-FFH,inplace=True) 
        df[idx] = df[idx] + FFH if flood_depth_only else df[idx]
        ax.semilogx(df_prb[idx], df[idx], label='{}: AAL {}'.format(idx,aal))
    plt.xlim(0.5,0.00001)
    y_lowerlim = 0 if flood_depth_only else -FFH
    y_maxlim = 1 if df[idx].max() <= 1 else df[idx].max()+(df[idx].max()*0.05)
    plt.ylim(y_lowerlim,y_maxlim)
    ax.legend()
    ax.grid(which='major', )
    ax.grid(which='minor', color='#4682B4', linestyle='--') #steelblue
    xtext = 'Flood Depth' if flood_depth_only else 'Flood Depth in Structure'
    ax.set_ylabel(f'{xtext}', fontsize=16)
    ax.set_xlabel('Probability of Occurence', fontsize=16)
    ax.set_title(f'{xtext} vs Probability', fontsize=20)
    pct_formatter = FuncFormatter(pct)
    for axis in [ax.xaxis]:
        axis.set_major_formatter(pct_formatter)
    return None

def PlotDamagesForStructure_MultipleIds(idxs: list,
                                        df_prb: pandas_DF,
                                        df_lss: pandas_DF,
                                        df_AAL: pandas_DF) -> mpl_figure:
    """For a Single Scenario, Multiple Structures"""
    fig, ax = plt.subplots(figsize=(20,5))
    for idx in idxs:
        df = df_lss.copy()
        aal = '${:,.2f}'.format(df_AAL.loc[idx,'AAL'])
        df[idx].fillna(0,inplace=True)
        ax.semilogx(df_prb[idx], df[idx], label='{}: AAL {}'.format(idx,aal))
    plt.xlim(0.5,0.00001)
    ax.grid(which='major', )
    ax.grid(which='minor', color='#4682B4', linestyle='--') #steelblue
    ax.legend()
    ax.set_ylabel(f'Damages', fontsize=16)
    ax.set_xlabel('Probability of Occurence', fontsize=16)
    ax.set_title(f'{idx}: Damages vs Probability', fontsize=20)
    pct_formatter = FuncFormatter(pct)
    usd_formatter = FuncFormatter(dollars)
    for axis in [ax.xaxis]:
        axis.set_major_formatter(pct_formatter)
    for axis in [ax.yaxis]:
        axis.set_major_formatter(usd_formatter)
    return None
    
def plotControlFileAALBoxPlots(df_aal: pd.DataFrame, root: pl.Path, ControlFilePath: pl.Path, 
                               drange: list, dscens: list, dnames_: list,
                               fs: int = 18, saveplot: bool = True) -> print:
    data, dnames = [], []
    for col in dscens:
        data.append(df_aal[col].values.tolist())
    drange = np.array(range(len(data))) + 1
    for i, d in enumerate(dnames_):
        d = d.replace('\\n','\n')
        agg = float(f'{np.sum(np.array(data[i])):0.2f}')
        med = float(f'{np.median(np.array(data[i])):0.2f}')
        avg = float(f'{np.mean(np.array(data[i])):0.2f}')
        d += f'\nSum:${agg:,}\nMed:${med:,}\nAvg:${avg:,}'
        dnames.append(d)
    fig, ax = plt.subplots(figsize = (2.5*len(data),7.5))
    ax.set_title(f"{ControlFilePath.name.replace('.xlsx','')}\nCompare Average Annual Loss Results", 
                 fontdict = {'size':str(fs)})
    ax.boxplot(data,)
    plt.ylabel('AAL ($)', fontdict = {'size':str(fs)})
    plt.xticks(drange, labels = dnames, rotation = 35, fontsize = fs-5) #rotation = 'vertical'
    if saveplot:
        fig.savefig(str(root/f"{ControlFilePath.name.replace('.xlsx','')}_boxplots.jpg"), bbox_inches = "tight")
    plt.show()
    return None