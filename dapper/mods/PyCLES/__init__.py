'''
Interface to models using PyCLES and common functionality for 2D states and 2 scalar time-constant parameters
'''

import numpy as np
# print('dpr')
import dapper as dpr
# print('plt')
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import shutil

data_dir = '/cluster/work/climate/dgrund/working_dir/dpr_data/data_Straka1993/'
# import dapper.mods as modelling
# import dapper.tools.liveplotting as LP
import dapper.tools.multiproc as multiproc

class pycles_model_config:
    '''Interface class.
    
    XXX
    '''

    def __init__(self, name, mp=False, plot_every_member=False, p=None):
        """Use `prms={}` to get the default configuration."""
        # Insert prms. Assert key is present in defaults.
        # infile_specs = default_infile_specs.copy()
        # for key in prms:
            # assert key in infile_specs
            # infile_specs[key] = prms[key]

        # self.infile_specs  = default_infile_specs
        # self.infile_base = make_infile(self.infile_specs)
        self.mp    = mp # False or int; set externally
        self.name  = name
        self.p = p if p else data_dir # adapt externally
        self.plot_every_member = plot_every_member # plot the data

        ## need to be set!
        # self.M = 1 # state components in E
        # self.P = 2 # parameter components in E

    def step_1(self, E_1, t, dt, dir):
        """Step a single state vector."""
        # assert self.infile_specs["t_max"] == dt # DA time step
        assert np.isfinite(t)           # final time

        # split
        assert len(E_1) == self.M+self.P
        x0 = E_1[:self.M]
        params = E_1[self.M:]

        x_t = self.call(x0, params, t, dt, dir, do_plot=self.plot_every_member)
        return x_t
    
    def step(self, E, t, dt):
        """Function needed for Dyn syntax: Dyn = {'model':model.step}
        Vector and 2D-array (ens) input, with multiproc for ens case."""
        if E.ndim == 1:
            print('[PyCLES.__init__.py] DAPPER-PyCLES in NON-PARALLEL MODE (one member only)')
            data_dir = self.make_data_dir()
            E = self.step_1(E, t, dt, data_dir)
            return E
        
        if E.ndim == 2:

            member_dirs = self.make_ensemble_dirs(N_ens=E.shape[0])

            if self.mp > 1:  # PARALLELIZED:
                print('[PyCLES.__init__.py] DAPPER-PyCLES in PARALLEL MODE')
                def call_step_1(n):
                    return self.step_1(
                        E[n], t=t, dt=dt, 
                        dir=member_dirs[n]
                    )
                print('[PyCLES.__init__.py] Using HMM.mp=',self.mp)
                with multiproc.Pool(self.mp) as pool:
                    E = pool.map(lambda x: call_step_1(x), [i for i in range(self.mp)])
                E = np.array(E)
            else:  # NON-PARALLELIZED:
                print('[PyCLES.__init__.py] DAPPER-PyCLES in NON-PARALLEL MODE')
                for n, x in enumerate(E):
                    print(f'Running ensemble member {n}.')
                    E[n] = self.step_1(
                        x, t, dt, 
                        member_dirs[n]
                    )
            print('[PyCLES.__init__.py] E.shape: ',E.shape)
            return E

    def make_ensemble_dirs(self, N_ens):
        dirs = os.listdir(self.p)
        dir_start = max([int(d) for d in dirs] + [0]) + 1
        member_dirs = [f'{self.p}{dir_start+n}/' for n in range(N_ens)]
        # data gen dir is the first (1 if self.p was empty)
        for p in member_dirs:
            os.mkdir(p)
        return member_dirs

    def make_data_dir(self):
        data_dir = self.p+'0/'
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        os.mkdir(data_dir)
        return data_dir

    def call(self, x0, params, t, dt):
        # XXX specify for each experiment
        pass



def load_xps(save_as):
    xps = dpr.xpList(dpr.load_xps(save_as))
    return xps

def print_summary(xps, Np, dists_prior):

    print('\n[PyCLES.__init__.py] Experiment summary:')
    print(xps.tabulate_avrgs([
        "rmse.state.a", "rmv.state.a",
        "rmse.param.a", "rmv.param.a",
    ]))

    xps_dists = get_xps_dists(xps, Np, dists_prior)

    for i,xp in enumerate(xps):
        dists = xps_dists[xp.name]
        print(f'\n{xp.name}:')
        for key in dists.keys():
            print(f'{key} = {list(dists[key])}')
        
    print('\nPRIOR')
    print(f'prior mean:\t{dists_prior["PRIOR_MEAN_PARAMS"]}')
    print(f'prior var:\t{dists_prior["PRIOR_VAR_PARAMS"]}')

    print('\nFORECAST') # taken from a random experiment
    print(f'mean:\t{dists["POST_MEAN_FOREC"]}')
    print(f'spread:\t{dists["POST_VAR_FOREC"]}')
    
    print('\nTRUTH')
    print(f'True params:\t{dists_prior["TRUE_PARAMS"]}')

def get_xps_dists(xps, Np, dists_prior):
    dists = {
        xp.name: get_xp_post_dists(xp,Np) for xp in xps
    }
    dists['prior'] = dists_prior

    return dists

def get_xp_post_dists(xp, Np):

    # These stats don't distinguish between parameters and state!
    m = np.squeeze(xp.stats.mu.a)
    s = np.squeeze(xp.stats.spread.a)

    mf = np.squeeze(xp.stats.mu.f)
    sf = np.squeeze(xp.stats.spread.f)

    return {
        'POST_MEAN':m[-Np:], # analysis
        'POST_VAR':s[-Np:],
        'POST_MEAN_FOREC':mf[-Np:],
        'POST_VAR_FOREC':sf[-Np:],
    }
    
def plot_norm(ax, mu, var, label, linestyle='-'):
    sigma = np.sqrt(var)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    c = 'b' if label=='prior' else 'r'
    ax.plot(x, stats.norm.pdf(x, mu, sigma),c, label=label)

def plot_dists_prior(*args, **kwargs):
    plot_dists_xps([],*args, post=False, **kwargs)

# missing: obs
# missing: case 1 xp
def plot_dists_xps(xps, dists_prior, plot_dir, Np, post=True):

    n_xps = len(xps) # test
    if n_xps == 0:
        fig, axs = plt.subplots(Np, figsize=(Np*4, 3))
        xps_dists = {
            'Priors':{}, # will be used as y label
            'prior':{}, # required formally
        }
        axs = [axs]
    elif n_xps == 1:
        fig, axs = plt.subplots(Np, figsize=(Np*4, 3))
        xps_dists = get_xps_dists(xps, Np,dists_prior) # dict
        axs = [axs]
    else:
        fig, axs = plt.subplots(n_xps, Np, figsize=(Np*4, n_xps*3))
        xps_dists = get_xps_dists(xps, Np,dists_prior) # dict
    j = 0
    for xp_name in xps_dists.keys():
        if xp_name == 'prior':
            j -= 1
        else:
            plot_dists_xp(axs[j], xps_dists[xp_name], dists_prior, Np, post=post, leftylabel=xp_name)
        j +=1
        
    plt.tight_layout()
    plt.savefig(plot_dir+'fig-dists.png')
    plt.show()

def plot_dists_xp(axs, dists, dists_prior, Np, post, leftylabel=''):

    # params
    for i in range(Np):
        plot_norm(axs[i],dists_prior['PRIOR_MEAN_PARAMS'][i],dists_prior['PRIOR_VAR_PARAMS'][0],'prior')
        if post:
            plot_norm(axs[i],dists['POST_MEAN'][i],dists['POST_VAR'][i],'posterior')
        axs[i].axvline(dists_prior['TRUE_PARAMS'][i],c='k', label='truth')
        axs[i].legend()
        axs[i].set_title(f'Parameter {i}')

    axs[0].set_ylabel(leftylabel)


def plot_dists_xps_onerow(xps, dists_prior, plot_dir, Np, post=True):

    fig, axs = plt.subplots(1, Np, figsize=(Np*4, 3))
    if len(xps) == 0:
        post = False
    else:
        xps_dists = get_xps_dists(xps, Np,dists_prior)

    # prior and truth
    for i,ax in enumerate(axs):
        plot_norm(ax,dists_prior['PRIOR_MEAN_PARAMS'][i],dists_prior['PRIOR_VAR_PARAMS'][0],'prior')
        ax.axvline(dists_prior['TRUE_PARAMS'][i],c='k', label='truth')
    
    # posterior(s)
    l = ['-','--','.']
    for j,xp_name in enumerate(xps_dists.keys()):
        if xp_name != 'prior' and post:
            dists = xps_dists[xp_name]
            for i,ax in enumerate(axs):
                plot_norm(ax,dists['POST_MEAN'][i],dists['POST_VAR'][i],xp_name,linestyle=l[j])
                ax.set_xlabel('parameter space')
                ax.set_title(f'Parameter {i}')
                ax.set_xlim((20,130))
                if i==0:
                    ax.set_ylabel('distribution')

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir+'fig-dists_onerow.png')
    plt.show()
