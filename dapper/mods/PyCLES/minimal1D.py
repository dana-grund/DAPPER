'''Minimal working example.'''

import numpy as np

import dapper as dpr
import dapper.mods as modelling
from dapper.mods.PyCLES import pycles_model_config
import dapper.tools.liveplotting as LP


############################
# Dynamics
############################

def call_dummy(x0, params, t, dt):
    # only one final measurement

    v,d = params[0], params[1]
    if type(x0)==np.ndarray:
        x = x0[0]
    else:
        x=x0

    x_t = np.array([( np.exp(v) - 0.2*d**3)])
    # x_t = np.array([(x + np.exp(v) - 0.2*d**3)])
    
    return np.hstack([x_t,params])


############################
# Model
############################

model = pycles_model_config(name="minimal1D", prms={})
Nx = model.M = 1
Np = model.P = 2
model.call = call_dummy

Dyn = {
    'M': model.M+model.P, # number of inferred components
    'model': model.step,
    'noise': 0,
}

dtout = model.infile_specs['dtout']
tseq = modelling.Chronology(dt=dtout, dko=1, T=dtout, BurnIn=0)


#############################
# Priors
#############################

# estimating state and parameters

TRUE_PARAMS = np.array([-1,2]) 
PRIOR_MEAN_PARAMS = np.array([0,0])
PRIOR_VAR_PARAMS = np.array([1**2])

TRUE_STATE = np.array([0])
PRIOR_MEAN_STATE = np.array([0])
PRIOR_VAR_STATE = np.array([1**2])

# OBS_VAR = np.array([0.1**2]) # set in minimal.py

def X0(param_mean, param_var):
    # State
    x0 = PRIOR_MEAN_STATE*np.ones(Nx)
    C0 = PRIOR_VAR_STATE*np.ones(Nx)
    # Append param params
    x0 = np.hstack([x0, param_mean*np.ones(Np)])
    C0 = np.hstack([C0, param_var*np.ones(Np)])
    return modelling.GaussRV(x0, C0)

def set_X0_and_simulate(hmm, xp):
    dpr.set_seed(3000)
    hmm.X0 = X0(TRUE_PARAMS, 0)
    xx, yy = hmm.simulate()
    hmm.X0 = X0(PRIOR_MEAN_PARAMS, PRIOR_VAR_PARAMS)
    return hmm, xx, yy


############################
# Observation settings
############################

# from Lorenz63/sakov2012
Nx = model.M # == len(x0)
Obs = modelling.Id_Obs(Nx) # observe full state
OBS_VAR = 0.1**2
Obs['noise'] = OBS_VAR  # modelling.GaussRV(C=CovMat(noise*eye(Nx))) ## overwritten later


############################
# Final model
############################

parts = dict(state=np.arange(model.M),
             param=np.arange(model.M)+model.P)

HMM = modelling.HiddenMarkovModel(
    Dyn, Obs, tseq, 
    sectors=parts,
    LP=LP.default_liveplotters,
)
