import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt

import sys
dir_add = '/cluster/work/climate/dgrund/git/dana-grund/doctorate_code/pycles/'
sys.path.insert(0,dir_add)
from plot_turb_stats import plot_scalar_stats_timeseries, plot_evol

def get_statsvar(OutDir,var='w_max'):
    # XXX get stats file name
    StatsDir = OutDir+'/stats/'
    print(f'{StatsDir=}')
    stats_file = os.listdir(StatsDir)[0] # the only file in there
    ds_ = xr.open_dataset(StatsDir+stats_file,group='timeseries')
    return ds_[var] # data_array


def plot_scalar_stats_timeseries(
    StatsFile,
    variables,
    SaveFolder,
    meas_locs,
):
    fig,axs = plt.subplots(len(variables),len(meas_locs), figsize=(3*len(meas_locs),3*len(variables)))

    for ax in axs.flatten():
        ax.grid()
        
    for i_var, var in enumerate(variables):
        for i_meas, meas_loc in enumerate(meas_locs):
            
            # if i_var==0 and i_meas==0:
            variable = f'{var}_loc{i_meas}'
            print(variable)
            da = load_stats_variable(StatsFile, variable, group='timeseries')
            print(da)
            da.plot(ax=axs[i_var, i_meas])
                
    # set titles of first row subplots as the variable names
    for i,variable in enumerate(variables):
        axs[i,0].set_ylabel(variable)

        # delete all other y labels
        for j in range(1,len(meas_locs)):
            axs[i,j].set_ylabel('')
                    
    # set titles of first column subplots as the measurement locations
    for i,meas_loc in enumerate(meas_locs):
        axs[0,i].set_title(f'{meas_loc}')
        
        # delete all but the bottom x labels
        for j in range(1,len(variables)):
            axs[j,i].set_xlabel('')
    
    # add vertical lines at t=0,300,600,900
    for ax in axs.flatten():
        ax.axvline(0, color='k', linestyle='--')
        ax.axvline(300, color='k', linestyle='--')
        ax.axvline(600, color='k', linestyle='--')
        ax.axvline(900, color='k', linestyle='--')
    
    save_file = 'fig-stats_timeseries'
    save_file = os.path.join(SaveFolder, save_file)
    save_figure(save_file)

def plot_evol(
    EnsDir,
    OutDir,
    SaveFolder,
    variable='temperature_anomaly',
    times=[0,300,600,900],
    cmap='turbo',
    title='',
    meas_locs=[],
):
    ds = load_all_timesteps(os.path.join(
        EnsDir,
        OutDir
    ))
        
    nt = len(times)    
    fig,axs = plt.subplots(nt, figsize=(6,2*nt))
    if nt==1: axs = [axs]
    
    da = ds[variable]
    vmin,vmax = float(da.min()), float(da.max())
    for it,t in enumerate(times):
        ax = axs[it]
        da_ = da.sel(t=t)
        da_.T.plot(ax=ax,cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_title(times[it])
        
        for loc in meas_locs:
            ax.plot(loc[0],loc[2],'kx')
        
    plt.suptitle(title)
    SaveName=f'fig-time_evol_{variable}.png'
    save_figure(os.path.join(
        SaveFolder,
        SaveName
    ))
       