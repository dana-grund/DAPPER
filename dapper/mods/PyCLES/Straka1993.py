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

np.random.seed(325)

############################
# Settings (modify)
############################

nx, nz = 256, 32    # resolution in pycles fixed to dx=dz=200
# Nx = 50             # length of the (dummy state and) obs in dapper -- obs_type=='hor_stripe'
# Nx = nx * nz        # length of the (dummy state and) obs in dapper -- obs_type=='full'
### Nx = Dyn.M-Np
Np = 2              # inferred parameters in dapper

specs = dict(
    # case_name           = 'StableBubble',
    v                   = 75, # default and truth
    d                   = 75, # default and truth
    r                   = 200, # should increase
    # t_max               = 20, # test
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
        print('Saved fig ',name)
        plt.show()
    else:
        print('bbb not plotting ',p_)
    return x_t.to_numpy()

def get_statsvar(OutDir):
    # XXX get stats file name
    StatsDir = OutDir+'/stats/'
    print(f'{StatsDir=}')
    stats_file = os.listdir(StatsDir)[0] # the only file in there
    ds_ = xr.open_dataset(StatsDir+stats_file,group='timeseries')
    return ds_['w_max'] # data_array

def obs_w_max(results_file, p_, do_plot=False):
    # maximal w in three pre-defined time windows
    # XXX needs to output sufficient dt
    # XXX not possiblt with -t, needs -T 900
    # Nx=3
    
    if do_plot: # full state
        x_t = get_ts_from_file(results_file)
        plot_xt(x_t,p_,name_add='full')
    
    FieldsDir, _ = os.path.split(results_file)
    print(f'{FieldsDir=}')
    OutDir = os.path.abspath(os.path.join(FieldsDir, os.pardir))
    print(f'{OutDir=}')
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
        print('Saved fig ',name)
        plt.show()
    else:
        print('bbb not plotting ',p_)
   
    return np.array(obs)

obs_funcs = {
    'full':         obs_full,
    'hor_stripe':   obs_hor_stripe,
    'w_max':        obs_w_max,
}

def call_obs_full(*args, **kwargs):
    return call(*args, obs_func=obs_funcs['full'], **kwargs)

def call_obs_hor_stripe(*args, **kwargs):
    return call(*args, obs_func=obs_funcs['hor_stripe'], **kwargs)

def call_obs_w_max(*args, **kwargs):
    return call(*args, obs_func=obs_funcs['w_max'], **kwargs)

call_funcs = {
    'full':         call_obs_full,
    'hor_stripe':   call_obs_hor_stripe,
    'w_max':        call_obs_w_max,
}

def get_call_func(obs_type): # need a func with same signature as call
    return call_funcs[obs_type]

Nx_by_obs = {
    'full':         nx * nz,
    'hor_stripe':   50,
    'w_max':        3,
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

    print('ENS_DIR (call()): ',p_)
    
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
    print('aaa plotting ',p_,': ',do_plot)
    
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
        shutil.copyfile(call_script,p_+'Straka1993_call.sh')
        
        # save_stdout = sys.stdout # XXX does not word
        # sys.stdout = open('pycles.out', 'x')
        os.system(f'bash {call_script}') # waits until execution is done
        # sys.stdout = save_stdout
        os.chdir(cwd)
        
        files = os.listdir(p_)
        for f in files:
            if f[:3] == 'Out':
                OutDir = f
            # if f[-3:] == '.in':
                # infile = f
        results_file = f'{p_}{OutDir}/fields/{specs_["t_max"]}.nc'
        
        return results_file

    def call_pycles_safely():
        results_file = call_pycles()
        print(f'Expected results_file: ',results_file)
        if os.path.exists(results_file):
            print(f'ZZZ PyCLES finished as expected in {p_=}')
        else:
            results_file = call_pycles()
            if os.path.exists(results_file):
                print(f'YYY PyCLES finished in second attempt in {p_=}')
            else:
                results_file = call_pycles()
                if os.path.exists(results_file):
                    print(f'YYY PyCLES finished in third attempt in {p_=}')
                else:
                    print(f'XXX PyCLES did not finish properly in {p_=}')
                    results_file = None
        return results_file
    
    print('Calling PyCLES...')
    results_file = call_pycles_safely()
    
    print('Reading output...')
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
    print('Nx in X0 ', Nx)
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

def create_HMM(mp=2, data_dir=None, obs_type='full'):

    Dyn = create_Dyn(p=data_dir, mp=mp, obs_type=obs_type)
    Nx = Nx_by_obs[obs_type]
    print('Nx in creat_hmm', Nx)
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
    print('Saved fig ',name)
    plt.show()