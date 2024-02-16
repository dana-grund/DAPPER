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
    # state estimation not used

    Mperdim = int(np.sqrt(Nx))
    v,d = params[0], params[1]

    # dyn
    a = np.arange(Mperdim)
    term1 = 0.01*a**2
    term2 = np.sin(a)
    term3 = np.cos(a)

    v_field = np.outer(term2,term3)
    d_field = np.outer(term3,term1)

    x_t = 0.5*v**2*v_field + 0.5*d**3*d_field

    # flatten
    x_t = x_t.ravel()

    return np.hstack([x_t,params])


############################
# Model
############################

model = pycles_model_config(name="minimal2D")
Nx = model.M = 100
# Nx = model.M = 25 # 5x5
Np = model.P = 2
model.call = call_dummy

Dyn = {
    'M': model.M+model.P, # number of inferred components
    'model': model.step,
    'noise': 0,
}
t_max = 10
tseq = modelling.Chronology(dt=t_max, dko=1, T=t_max, BurnIn=0)


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
from dapper.mods.utils import name_func, ens_compatible
def Max_Obs(Nx): # XXX not working
    """Scalar max observable

    It is not a function of time.

    Parameters
    ----------
    Nx: int
        Length of state vector
    obs_inds: ndarray
        The observed indices.

    Returns
    -------
    Obs: dict
        Observation operator including size of the observation space,
        observation operator/model and tangent linear observation operator
    """
    Ny = 2
    @name_func(f"State maximum observable")
    @ens_compatible
    def model(x): return np.vstack([np.max(x[:Nx]),np.min(x[:Nx])])
    # @name_func(f"Constant matrix\n{H}")
    # def linear(x): return H
    Obs = {
        'M': Ny,
        'model': model,
        # 'linear': linear,  # XXX DOES NOT SEEM TO WORK WITHOUT
    }
    return Obs

# from Lorenz63/sakov2012
np.random.seed(325)
obs_idcs = np.random.choice(np.arange(Nx),size=3)
# print('Observing the following field indices: ',obs_idcs)
Obs = modelling.partial_Id_Obs(Nx,obs_idcs) # observe full state 
# Obs = modelling.Id_Obs(Nx) # observe full state 
# Obs = Max_Obs(Nx)
OBS_VAR = 0.001**2
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

############################
# Summary
############################

dists_prior = {
    'PRIOR_MEAN_PARAMS':PRIOR_MEAN_PARAMS,
    'PRIOR_VAR_PARAMS':PRIOR_VAR_PARAMS,
    'PRIOR_MEAN_STATE':PRIOR_MEAN_STATE,
    'PRIOR_VAR_STATE':PRIOR_VAR_STATE,
    'OBS_VAR':OBS_VAR,
    'TRUE_PARAMS':TRUE_PARAMS,
    'TRUE_STATE':TRUE_STATE,
}