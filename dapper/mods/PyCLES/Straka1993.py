import numpy as np
import os 
import sys
import xarray as xr
import json
import shutil
import pdb

import dapper.mods as modelling
import dapper.tools.liveplotting as LP
from dapper.mods.PyCLES import * # pycles_model_config, data_dir

import sys
dir_add = '/cluster/work/climate/dgrund/git/dana-grund/doctorate_code/pycles/'
sys.path.insert(0,dir_add)
from plot_turb_stats import plot_scalar_stats_timeseries, plot_evol
np.random.seed(325)

############################
# Settings (modify)
############################

nx, nz = 256, 32    # resolution in pycles fixed to dx=dz=200
# Nx = 50             # length of the (dummy state and) obs in dapper -- obs_type=='hor_stripe'
# Nx = nx * nz        # length of the (dummy state and) obs in dapper -- obs_type=='full'
### Nx = Dyn.M-Np
Np = 2              # inferred parameters in dapper # SGS v and d

specs = dict(
    # case_name           = 'StableBubble',
    v                   = 75, # default and truth
    d                   = 75, # default and truth
    r                   = 200, # should increase
    t_max               = 900, 
    p                   = data_dir,
    # variable            = 'w',
)
t_max = specs['t_max']

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

def get_ts_from_file(results_file):
    '''a single time step in results_file, indexed ith it=0'''
    ds = xr.open_dataset(results_file)
    return ds['w'].isel(y=2).isel(t=0) # XXX w hard-coded
    
def obs_full(results_file, p_, do_plot=False):
    x_t = get_ts_from_file(results_file)
    if do_plot:
        plot_xt(x_t,p_,name_add='full')
    return x_t.to_numpy()

def obs_hor_stripe(results_file, p_, do_plot=False):
    # point observation within a horizontal stripe through the rotors 
    # at 1km height, within the x range of interest (km 30-40, right half)
    # XXX fixed for resolution dx=dz=200 m
    # Nx = 50
    
    x_t = get_ts_from_file(results_file)
    if do_plot:
        plot_xt(x_t,p_,name_add='full')
    xmin,xmax = 150,200
    x_t = x_t.isel(z=5).isel(x=slice(xmin,xmax))
    if do_plot:
        plt.figure(figsize=(8,5))
        x_half = 128
        x_space = np.arange(xmin-x_half,xmax-x_half,1)
        # x_space = np.arange(30,36,0.2) # km
        plt.plot(x_space, x_t, '-x')
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
    return x_t.to_numpy()

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
    # Nx=3
    
    if do_plot: # full state
        x_t = get_ts_from_file(results_file)
        plot_xt(x_t,p_,name_add='full')
    
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
    # Nx = XXX
    
    T = 90
    # T = 900
    dt = 10
    Ntimes = T//dt + 1 # incl t=0 # 91
    # variables = ['turb_meas_u','turb_meas_w','turb_meas_theta']
    variables = ['turb_meas_theta']
    Nvars = len(variables)
    meas_locs = [
        [10000, 0, 500],
        [10000, 0, 2000],
        [20000, 0, 500],
        [20000, 0, 2000],
    ]
    Nlocs = len(meas_locs)
    Nx = Ntimes * Nvars * Nlocs # 1092
        
    if do_plot: # full state
        x_t = get_ts_from_file(results_file)
        plot_xt(x_t,p_,name_add='full')
    
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
    else:
        print('bbb not plotting ',p_)
        
    return np.array(time_series).ravel()    
    

obs_funcs = {
    'full':         obs_full,
    'hor_stripe':   obs_hor_stripe,
    'w_max':        obs_w_max,
    'timeseries':   obs_timeseries,
}

def call_obs_full(*args, **kwargs):
    return call(*args, obs_func=obs_funcs['full'], **kwargs)

def call_obs_hor_stripe(*args, **kwargs):
    return call(*args, obs_func=obs_funcs['hor_stripe'], **kwargs)

def call_obs_w_max(*args, **kwargs):
    return call(*args, obs_func=obs_funcs['w_max'], **kwargs)

def call_obs_timeseries(*args, **kwargs):
    return call(*args, obs_func=obs_funcs['timeseries'], **kwargs)

call_funcs = {
    'full':         call_obs_full,
    'hor_stripe':   call_obs_hor_stripe,
    'w_max':        call_obs_w_max,
    'timeseries':   call_obs_timeseries,
}

def get_call_func(obs_type): # need a func with same signature as call
    return call_funcs[obs_type]

Nx_by_obs = {
    'full':         nx * nz,
    'hor_stripe':   50,
    'w_max':        3,
    'timeseries':   10, # 1092,
}

############################
# Dynamics
############################

def write_to_bash_variables(dir,specs):
    VARS = dict(
        # case_name           = 'CASE_NAME',
        v                   = 'V',
        d                   = 'D',
        r                   = 'R',
        t_max               = 'T',
        p                   = 'ENS_DIR',
        variable            = 'variable',
    )
    for var in specs.keys():
        text = f'{VARS[var]}={specs[var]}\n'
        with open(dir+'parameters.sh','a+') as f:
            f.write(text)

def call(x0, params, t, dt, p_, do_plot=False, obs_func=obs_full):

    v,d = params

    print('[Straka1993.py] calling on ENS_DIR = ',p_)
    
    specs_ = specs.copy()
    specs_.update({
        'v':v,
        'd':d,
        'p':p_,
    })

    call_script = '/cluster/work/climate/dgrund/git/dana-grund/DAPPER/dapper/mods/PyCLES/Straka1993_call.sh'
    
    # if do_plot:
    # plot the first 5 members (incl. data)
    do_plot = p_[-3]=='/' and int(p_[-2])<5
    
    def call_pycles():
        cwd = os.getcwd()
        os.chdir(p_)

        # delete content in case repeted run  # XXX only outdir!
        for root, dirs, files in os.walk(p_):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        write_to_bash_variables(p_,specs_)
        
        os.system(f'bash {call_script}') # waits until execution is done
        os.chdir(cwd)
        
        files = os.listdir(p_)
        OutDir = None
        for f in files:
            if f[:3] == 'Out':
                OutDir = f
        assert OutDir is not None, "Could not find OutDir!"

        results_file = f'{p_}{OutDir}/fields/{specs_["t_max"]}.nc'
        assert os.path.exists(results_file), f"Expected PyCLES results_file does not exist: {results_file}"
        
        return results_file

    results_file = call_pycles()
    print(f'[Straka1993.py] Found results_file: ',results_file)
    
    # print('Reading output...')
    if results_file is not None:
        x_t = obs_func(results_file, p_, do_plot)        
    else:
        x_t = np.zeros_like(x0)

    ts_extended = np.concatenate([x_t.ravel(),params])
    
    return ts_extended # 1D array

############################
# Model
############################

def create_Dyn(p, mp, obs_type):
    model = pycles_model_config(name="Straka1993", mp=mp, p=p)
    model.M = Nx_by_obs[obs_type]
    model.P = Np
    model.call = get_call_func(obs_type) # XXX TEST
    # model.call = call

    Dyn = {
        'M': model.M+model.P, # number of inferred components
        'model': model.step,
        'noise': 0,
    }

    return Dyn

#############################
# Setup and prior
#############################

def X0(param_mean, param_var, Nx):
    # State
    x0 = PRIOR_MEAN_STATE*np.ones(Nx)
    C0 = PRIOR_VAR_STATE*np.ones(Nx)
    # Append param params
    x0 = np.hstack([x0, param_mean*np.ones(Np)])
    C0 = np.hstack([C0, param_var*np.ones(Np)])
    return modelling.GaussRV(x0, C0)

def set_X0_and_simulate(hmm, xp):
    dpr.set_seed(3000)
    Nx = hmm.Dyn.M-Np
    hmm.X0 = X0(TRUE_PARAMS, 0, Nx)
    xx, yy = hmm.simulate() # perfect model data simulation
    hmm.X0 = X0(PRIOR_MEAN_PARAMS, PRIOR_VAR_PARAMS, Nx)
    return hmm, xx, yy

############################
# Observation settings
############################

def create_Obs(Nx):
    # obs_idcs = np.random.choice(np.arange(Nx),size=3)
    # print('Observing the following field indices: ',obs_idcs)
    # Obs = modelling.partial_Id_Obs(Nx,obs_idcs) # observe partial state 
    Obs = modelling.Id_Obs(Nx) # observe full state 
    Obs['noise'] = OBS_VAR  # modelling.GaussRV(C=CovMat(noise*eye(Nx))) ## overwritten later

    return Obs

############################
# Final model
############################

def create_HMM(mp=1, data_dir=None, obs_type='full'):

    Dyn = create_Dyn(p=data_dir, mp=mp, obs_type=obs_type)
    Nx = Nx_by_obs[obs_type]
    Obs = create_Obs(Nx)

    tseq = modelling.Chronology(dt=t_max, dko=1, T=t_max, BurnIn=0)
    
    parts = dict(state=np.arange(Nx),
                param=np.arange(Nx)+Np)

    HMM = modelling.HiddenMarkovModel(
        Dyn, Obs, tseq, 
        sectors=parts,
        LP=LP.default_liveplotters,
    )

    return HMM

############################
# Plotting
############################

def plot_xt(x_t,plot_dir,name_add=''):
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