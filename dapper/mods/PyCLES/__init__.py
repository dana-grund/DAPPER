'''Interface to models using PyCLES.'''

import matplotlib as mpl
import numpy as np

import dapper.mods as modelling
import dapper.tools.liveplotting as LP

# import pycles # XXX

#########################
# Model
#########################

default_infile_specs = dict(
    viscosity_list      = [75],
    diffusivity_list    = [75],
    resolution          = 200,
    t_max               = 100,
    path                = './data/', # XXX ADAPT DATA DIR
)
'''Straka96 key words:
    viscosity_list=None,
    diffusivity_list=None,
    resolution_list=None,
    vd_mode=None,
    v_eq_d=False,
    IC_lists={},
    ic_mode=None,
    members_list=None,
    t_max=None,
    nproc_list=None,
    test=False,
    path='./'
'''

class pycles_model_config:
    '''Interface class.
    
    XXX
    '''

    def __init__(self, name, mp=False):
        """Use `prms={}` to get the default configuration."""
        # Insert prms. Assert key is present in defaults.
        # infile_specs = default_infile_specs.copy()
        # for key in prms:
            # assert key in infile_specs
            # infile_specs[key] = prms[key]

        self.infile_specs  = default_infile_specs
        # self.infile_base = make_infile(self.infile_specs)
        self.mp    = mp
        self.name  = name

        ## need to be set!
        # self.M = 1 # state components in E
        # self.P = 2 # parameter components in E

    def step_1(self, E_1, t, dt):
        """Step a single state vector."""
        assert self.infile_specs["dtout"] == dt # DA time step
        assert np.isfinite(t)           # final time

        # split
        assert len(E_1) == self.M+self.P
        x0 = E_1[:self.M]
        params = E_1[self.M:]

        # # make new infile
        # infile_path = update_infile(self.infile_base, params, t, dt)
        # return call_pycles(infile_path)

        x_t = self.call(x0, params, t, dt)
        return x_t
    
    def step(self, E, t, dt):
        """Vector and 2D-array (ens) input, with multiproc for ens case."""
        if E.ndim == 1:
            return self.step_1(E, t, dt)
        if E.ndim == 2:
            # NON-PARALLELIZED:
            for n, x in enumerate(E):
                E[n] = self.step_1(x, t, dt)
            return E

    def call(self, x0, params, t, dt):
        # XXX specify for each experiment
        pass


def call_pycles(infile_path):
    # call pycles XXX
    # read output to numpy XXX
    x_t = None
    return x_t

def make_infile(infile_specs):
    # XXX

    # call infile generator
    # save infile
    infile_full = infile_specs
    infile_path = ''

    return infile_full, infile_path

def update_infile(infile_base, params, t, dt):
    # XXX
    pass