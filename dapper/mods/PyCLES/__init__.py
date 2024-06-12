'''
Interface to models using PyCLES and common functionality for 2D states and 2 scalar time-constant parameters
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import shutil
import time

default_dir = '/cluster/work/climate/dgrund/working_dir/dpr_data/data_Straka1993/'
# import dapper.mods as modelling
# import dapper.tools.liveplotting as LP
# import dapper.tools.multiproc as multiproc

class PyCLES_interface:
    '''Interface class.
    
    XXX
    '''

    def __init__(
        self, 
        name,
        # mp=False, 
        plot_every_member=False, 
        data_dir=None,
        obs_func=None, 
        t_max=900,
        dx=200, 
        No=None, 
        Np=None
    ):
        """submits a job for each member, so automatically parallelizing, no mp keyword needed"""

        # PyCLES
        self.name  = name
        self.plot_every_member = plot_every_member # plot the data
        self.obs_func = obs_func
        self.t_max = t_max
        self.dx = dx
        
        # DAPPER
        assert No is not None and Np is not None, "No and Np need to be set!"
        # self.Nx = Nx # state ### not used
        self.No = No # observations
        self.Np = Np # parameters
        self.M = self.No + self.Np # extended state as seen by DAPPER
        
        # directories
        self.data_dir = data_dir if data_dir else default_dir # adapt externally
        self.make_truth_dir()
        self.member_dirs = [] # will be set in step

    def start_simulation(
        self, E_1, t, dt, dir
    ):        
        # --- split state and parameters
        assert len(E_1) == self.M
        x0 = E_1[:self.No]
        params = E_1[self.No:]
        
        v,d = params
        specs = {
            'v':v,
            'd':d,
            'p':dir,
            'r':self.dx,
            't_max':self.t_max,
        }

        # --- simulate
        submit_pycles_job(specs)
        return params
        
    def get_result_and_observe(self, dir, params):
        

        # --- wait for computation to finish
        max_waiting_time = 360 # s
        waiting_interval = 3 # s
        time_elapsed = 0 # s
        results_file = None
        
        while results_file is None:
            time.sleep(3)
            time_elapsed += waiting_interval
            if time_elapsed > max_waiting_time:
                print('[PyCLES.__init__.py] Timeout while waiting for result in ',dir)
                break
            
            results_file = get_results_file(dir,self.t_max)
    
        print(f'[PyCLES.__init__.py] Found results_file: ',results_file)
        
        # --- observe
        obs_t = self.obs_func(results_file, dir, self.plot_every_member)        

        # --- concatenate state
        extended_state = np.concatenate([obs_t.ravel(),params])
        return extended_state # 1D array

    def step(self, E, t, dt):
        """Function needed for Dyn syntax: Dyn = {'model':model.step}
        Vector and 2D-array (ens) input, with multiproc for ens case."""
        if E.ndim == 1:
            ### assumes only the data generation is called single!!
            print(f'[PyCLES.__init__.py] Starting single simulation on dir=self.truth_dir={self.truth_dir}.')
            params = self.start_simulation(E, t=t, dt=dt, dir=self.truth_dir)
            E = self.get_result_and_observe(self.truth_dir, params)
            
            return E
        
        if E.ndim == 2:

            self.member_dirs = self.make_member_dirs(N_ens=E.shape[0])
            
            params_list = []
            for n,dir in enumerate(self.member_dirs):
                params = self.start_simulation(E[n], t=t, dt=dt, dir=dir)
                params_list.append(params)
            
            E = []
            for n,dir in enumerate(self.member_dirs):
                E.append(
                    self.get_result_and_observe(dir, params_list[n])
                )
            E = np.array(E)
                
            return E

    def make_member_dirs(self, N_ens):
        '''The data generated by the ensemble members.'''
        dirs = os.listdir(self.data_dir)
        dir_start = max([int(d) for d in dirs] + [0]) + 1
        member_dirs = [f'{self.data_dir}{dir_start+n}/' for n in range(N_ens)]
        # data gen dir is the first (1 if self.data_dir was empty)
        for p in member_dirs:
            os.mkdir(p)
        print(f'[PyCLES.__init__.py] Made {len(member_dirs)} member dirs starting with {member_dirs[0]}')
        return member_dirs

    def make_truth_dir(self):
        '''The true data.'''
        truth_dir = self.data_dir+'0/'
        if os.path.isdir(truth_dir):
            shutil.rmtree(truth_dir)
        os.mkdir(truth_dir)
        print(f'[PyCLES.__init__.py] Made truth dir {truth_dir}')
        self.truth_dir = truth_dir
    
def get_results_file(dir,t_max):
    files = os.listdir(dir)
    OutDir = None
    for f in files:
        if f[:3] == 'Out':
            OutDir = f
    if OutDir is None:
        # computation did not start yet
        return None
    else:
        results_file = f'{dir}{OutDir}/fields/{t_max}.nc'
        if os.path.exists(results_file):
            # computation finished
            return results_file
        else:
            # computation did not finish yet
            return None

    
def submit_pycles_job(specs):
    '''Submits the sample as a job but does not wait for completion'''
    call_script = '/cluster/work/climate/dgrund/git/dana-grund/DAPPER/dapper/mods/PyCLES/Straka1993_call_job.sh'
    p = specs['p']
    cwd = os.getcwd()
    os.chdir(p)

    # delete content in case repeted run  # XXX only outdir!
    for root, dirs, files in os.walk(p):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    write_to_bash_variables(p,specs)
    
    os.system(f'bash {call_script} {p}')
    os.chdir(cwd)

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


def plot_dists_xps(xps, dists_prior, plot_dir, Np, post=True):
    # missing: obs
    # missing: case 1 xp
    
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
