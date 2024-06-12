import numpy as np
import os 
import xarray as xr

import dapper.mods as modelling
# import dapper.tools.liveplotting as LP
from dapper.mods.PyCLES import PyCLES_interface

############################
### to be replaced by stand-alone code or other package
import sys
dir_add = '/cluster/work/climate/dgrund/git/dana-grund/doctorate_code/pycles/'
sys.path.insert(0,dir_add)
from plot_turb_stats import plot_scalar_stats_timeseries, plot_evol
### to be replaced by stand-alone code or other package
############################


############################
# Settings (modify)
############################

### No = size of the extended model state in DAPPER
### nx = resolution (number of grid points)in PyCLES

nx, nz = 256, 32    # resolution in pycles fixed to dx=dz=200
# No = 50             # length of the (dummy state and) obs in dapper -- obs_type=='hor_stripe'
# No = nx * nz        # length of the (dummy state and) obs in dapper -- obs_type=='full'
### No = Dyn.M-Np
Np = 2              # inferred parameters in dapper # SGS v and d

# estimating state and parameters
TRUE_PARAMS = np.array([75,75]) 
PRIOR_MEAN_PARAMS = np.array([50,100])
PRIOR_VAR_PARAMS = np.array([10**2])

TRUE_STATE = np.array([0])
PRIOR_MEAN_STATE = np.array([0])
PRIOR_VAR_STATE = np.array([1**2])

OBS_VAR = 0.1**2 # XXX change

dists_prior = {
    'PRIOR_MEAN_PARAMS':PRIOR_MEAN_PARAMS,
    'PRIOR_VAR_PARAMS':PRIOR_VAR_PARAMS,
    'PRIOR_MEAN_STATE':PRIOR_MEAN_STATE,
    'PRIOR_VAR_STATE':PRIOR_VAR_STATE,
    'OBS_VAR':OBS_VAR,
    'TRUE_PARAMS':TRUE_PARAMS,
    'TRUE_STATE':TRUE_STATE,
}

############################
# Equip dynamics with observable
############################

def get_field_from_file(results_file):
    '''a single time step in results_file, indexed ith it=0'''
    ds = xr.open_dataset(results_file)
    return ds['w'].isel(y=2).isel(t=0) # XXX w hard-coded
    
def obs_full(results_file, p_, do_plot=False):
    x_t = get_field_from_file(results_file)
    if do_plot:
        plot_field(x_t,p_,name_add='full')
    return x_t.to_numpy()

def obs_hor_stripe(results_file, p_, do_plot=False):
    # point observation within a horizontal stripe through the rotors 
    # at 1km height, within the x range of interest (km 30-40, right half)
    # XXX fixed for resolution dx=dz=200 m
    # No = 50
    
    x_t = get_field_from_file(results_file)
    if do_plot:
        plot_field(x_t,p_,name_add='full')
    xmin,xmax = 150,200
    
    # obs
    obs = x_t.isel(z=5).isel(x=slice(xmin,xmax))
    
    if do_plot:
        plt.figure(figsize=(8,5))
        x_half = 128
        x_space = np.arange(xmin-x_half,xmax-x_half,1)
        # x_space = np.arange(30,36,0.2) # km
        plt.plot(x_space, obs, '-x')
        plt.xlabel('x')
        plt.ylabel('w')
        plt.title('data x_t at final time (z=1km)')
        plt.tight_layout()
        name = f'{p_}fig-data_hor_slice.png'
        plt.savefig(name)
        print('Saved figure ',name)
        plt.show()
    else:
        # print('bbb not plotting ',p_)
        pass
    return obs.to_numpy()

def get_statsfile(OutDir):
    StatsDir = OutDir+'/stats/'
    return StatsDir + os.listdir(StatsDir)[0] # the only file in there
    
def get_statsvar(OutDir,var='w_max'):
    stats_file = get_statsfile(OutDir)
    ds_ = xr.open_dataset(stats_file,group='timeseries')
    return ds_[var] # data_array

def obs_w_max(results_file, p_, do_plot=False):
    # maximal w in three pre-defined time windows
    # XXX needs to output sufficient dt
    # XXX not possiblt with -t, needs -T 900
    # No=3
    
    if do_plot: # full state
        x_t = get_field_from_file(results_file)
        plot_field(x_t,p_,name_add='full')
    
    FieldsDir, _ = os.path.split(results_file)
    OutDir = os.path.abspath(os.path.join(FieldsDir, os.pardir))
    time_series = get_statsvar(OutDir) 
    obs = []
    times = []
    for (tmin,tmax) in [(0,400), (400,700), (700,900)]:
        ts_ = time_series.copy()
        ts_n = ts_.to_numpy()
        time_space = ts_['t'].to_numpy()
        wmax = float(ts_.sel(t=slice(tmin,tmax)).max())
        obs.append(wmax)
        times.append(time_space[np.where(ts_n==wmax)])
    
    if do_plot:
        plt.figure(figsize=(8,5))
        time_series.plot(c='k')
        plt.scatter(times,obs,marker='*',s=150,c='k')

        plt.axvline(0, c='darkgray', linestyle=':')
        plt.axvline(300, c='darkgray', linestyle=':')
        plt.axvline(600, c='darkgray', linestyle=':')
        plt.axvline(900, c='darkgray', linestyle=':')
        plt.axvline(400, c='darkgray', linestyle='-')
        plt.axvline(700, c='darkgray', linestyle='-')
        
        plt.xlabel(r'$t~~[s]$')
        plt.ylabel(r'$u_{max},~w_{max}~~[m/s]$')

        plt.tight_layout()
        name = f'{p_}fig-data_wmax.png'
        plt.savefig(name)
        print('Saved figure ',name)
        plt.show()
   
    return np.array(obs)

def obs_timeseries(results_file, p_, do_plot=False):
    # time series of u,w,T at 8 predefined locations (see Straka1993_call.sh)
    # fied output time step of 10 s
    # needs -T 900
    # No = XXX
    

    # variables = ['turb_meas_u','turb_meas_w','turb_meas_theta']
    variables = ['turb_meas_theta']
    meas_locs = [
        [10000, 0, 500],
        [10000, 0, 2000],
        [20000, 0, 500],
        [20000, 0, 2000],
    ]
    
    # T = 900 ### XXX hard-coded!
    # dt = 10
    # Ntimes = T//dt + 1 # incl t=0 # 91
    # Nvars = len(variables)
    # Nlocs = len(meas_locs)
    # No = Ntimes * Nvars * Nlocs # 1092
        
    if do_plot: # full state
        x_t = get_field_from_file(results_file)
        plot_field(x_t,p_,name_add='full')
    
    FieldsDir, _ = os.path.split(results_file)
    OutDir = os.path.abspath(os.path.join(FieldsDir, os.pardir))
    time_series = []
    for v in variables:
        # for loc in range(4):
        for loc in range(1):
            var = f'{v}_loc{loc}'
            time_series.append(get_statsvar(OutDir,var=var).to_numpy())
    
    if do_plot:
        StatsFile = get_statsfile(OutDir)
        plot_scalar_stats_timeseries(
            StatsFile,
            variables,
            p_,
            meas_locs,
        )
        plot_evol(
            OutDir,
            p_,
            variable='temperature', # temperature_anomaly
            times=[0,T], # add more
            cmap='turbo',
            title='',
            meas_locs=[],
        )
        
    return np.array(time_series).ravel()    
    

obs_funcs = {
    'full':         obs_full,
    'hor_stripe':   obs_hor_stripe,
    'w_max':        obs_w_max,
    'timeseries':   obs_timeseries,
}

def No_by_obs(obs_type, t_max=None, dt=10, n_vars=1):
    defaults = {
        'full':         nx * nz,
        'hor_stripe':   50,
        'w_max':        3,
    }
    if obs_type in defaults:
        return defaults[obs_type]
    elif obs_type == 'timeseries':
        # 91 for dt=10, T=900, nvars=1
        return (t_max//dt + 1) * n_vars

############################
# Dynamics
############################

def create_Dyn(data_dir,obs_type, t_max, dx=50, No=None):
    obs_func = obs_funcs[obs_type]
    model = PyCLES_interface(
        name="Straka1993",
        # mp=mp, 
        data_dir=data_dir, 
        obs_func=obs_func,
        t_max=t_max,
        dx=dx,
        No=No,
        Np=Np,
    )
    Dyn = {
        'M': model.M, # extended state (observations, parameters)
        'model': model.step,
        'noise': 0,
    }

    return Dyn

#############################
# Setup and prior
#############################

def X0(param_mean, param_var, No):
    # State --> is actually obs here!!!
    x0 = PRIOR_MEAN_STATE*np.ones(No)
    C0 = PRIOR_VAR_STATE*np.ones(No)
    # Append param params
    x0 = np.hstack([x0, param_mean*np.ones(Np)])
    C0 = np.hstack([C0, param_var*np.ones(Np)])
    return modelling.GaussRV(x0, C0)

def set_X0_and_simulate(hmm, xp):
    No = hmm.Dyn.M-Np
    hmm.X0 = X0(TRUE_PARAMS, 0, No)
    xx, yy = hmm.simulate() # perfect model data simulation
    hmm.X0 = X0(PRIOR_MEAN_PARAMS, PRIOR_VAR_PARAMS, No)
    return hmm, xx, yy

############################
# Observation settings
############################

def create_Obs(No):
    
    # obs_idcs = np.random.choice(np.arange(No),size=3)
    # print('Observing the following field indices: ',obs_idcs)
    # Obs = modelling.partial_Id_Obs(No,obs_idcs) # observe partial state 
    
    Obs = modelling.Id_Obs(No) # observe full state. 
    # Here used to "observe" the full output by PyCLES_interface, which is the observation already.
    Obs['noise'] = OBS_VAR  # modelling.GaussRV(C=CovMat(noise*eye(No))) ## overwritten later

    return Obs

############################
# Final model
############################

def create_HMM(data_dir=None, obs_type='full', t_max=900, dx=50):

    No = No_by_obs(obs_type, t_max=t_max)
    Dyn = create_Dyn(data_dir=data_dir,obs_type=obs_type, t_max=t_max, dx=dx, No=No)
    Obs = create_Obs(No)

    tseq = modelling.Chronology(dt=t_max, dko=1, T=t_max, BurnIn=0)
    
    parts = dict(state=np.arange(No),
                param=np.arange(No)+Np)

    HMM = modelling.HiddenMarkovModel(
        Dyn, Obs, tseq, 
        sectors=parts,
        # LP=LP.default_liveplotters,
    )

    return HMM

############################
# Plotting
############################

def plot_field(x_t,plot_dir,name_add=''):
    # full field
    
    field = x_t.reshape((nx,nz)) if len(x_t.shape) == 1 else x_t
    # x_space = np.arange(25.6,36,0.2) # km
    # z_space = np.arange(0,6.4,0.2) # km
    
    plt.figure(figsize=(8,5))
    plt.imshow(field[nx//2:,:].T, origin='lower')
    plt.colorbar()
    # plt.xlabel('x [km]')
    # plt.xlabel('z [km]')
    plt.title('data x_t at final time')
    plt.tight_layout()
    name = f'{plot_dir}fig-data_{name_add}.png'
    plt.savefig(name)
    print('Saved figure ',name)
    plt.show()