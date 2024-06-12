import time
s = time.time()

print('[demo_Straka1993.py] Importing...')
import os
from mpl_tools import is_notebook_or_qt as nb
import argparse
import numpy as np
t = time.time()
print(f'[demo_Straka1993.py] Importing packages took {t-s:.2f} seconds.')

from dapper import set_seed, xpList, load_xps, xpSpace
tt = time.time()
print(f'[demo_Straka1993.py] Importing dapper took {tt-t:.2f} seconds.')

import dapper.da_methods as da
t = time.time()
print(f'[demo_Straka1993.py] Importing da_methods took {t-tt:.2f} seconds.')

import dapper.mods.PyCLES.Straka1993 as S93
tt = time.time()
print(f'[demo_Straka1993.py] Importing Straka1993 took {tt-t:.2f} seconds.')

from dapper.mods.PyCLES import print_summary, plot_dists_prior, plot_dists_xps_onerow, plot_field
t = time.time()
print(f'[demo_Straka1993.py] Importing PyCLES took {t-tt:.2f} seconds.')

print('[demo_Straka1993.py] Importing done.')

np.random.seed(325)
set_seed(3000)

def set_up_experiments(N_ens):
    xps = xpList()
    # xps += PartFilt(N=1000, reg=1)
    ''' On the ensemble size
    - N>= 2 as we need N-1 for variances
    - One model eval more used for data generation in set_X0_and_simulate()
    '''
    xps += da.EnKF('Sqrt', N_ens, infl=1.04)
    # xps += da.EnKF_N(N_ens, xN=2) # xN for hyperprior coeffs
    # xps += da.iEnKS("Sqrt", N_ens, Lag=1, xN=2, nIter=5, wtol=1e-5)
    return xps

def launch_experiments(HMM,xps):
    # HMM.mp = N_ens # one process per member # does not work here! set manually.
    # XXX caution with large ensembles!
    scriptname = HMM.name if nb else __file__
    save_as = xps.launch(
        HMM, scriptname, setup=S93.set_X0_and_simulate,
        # mp=False,           # Multiprocessing
        fail_gently=True,  # Facilitate debugging
        liveplots=False,    # NB: Turn off if running iEnKS
        free=False,         # Don't delete time series (enables replay)
    )
    return save_as

def plot_obs_examples(HMM,xps,dir):
    HMM, xx, yy = S93.set_X0_and_simulate(HMM,xps[0])
    plot_field(yy[0],dir)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(prog='')
    parser.add_argument('-p','--path', type=str, default='./',
                        help='Where to store figs and data in folder p/data/')
    parser.add_argument('-o','--obs_type', type=str, default='full',
                        help='Type ob obervations, see Straka1993.py')
    parser.add_argument('-N','--N_ens', type=int, default=20,
                        help='Number of ensemble members (>=2)')
    # parser.add_argument('-T','--T', type=int, default=900, required=False,
                        # help='Final simulation time')
    parser.add_argument('--dx', type=int, default=200, required=False,
                        help='Grid spacing of the model (m)')
    # parser.add_argument('--mp', type=int, default=0, required=False,
                        #  help='Number of parallel member computations. Default: mp=N_ens.')
    args = parser.parse_args()
    
    args.T = 900 # hard-coded in observations!
    
    dir = args.path
    data_dir = dir + 'data/'
    plot_dir = dir + 'plots/'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    
    N_ens = args.N_ens
    obs_type = args.obs_type
    # if args.mp == 0:
    #     args.mp = N_ens

    print('Call create_HMM...')
    HMM = S93.create_HMM(data_dir=data_dir, obs_type=obs_type, t_max=args.T, dx=args.dx)

    print('Call set_up_experiments...')
    xps = set_up_experiments(N_ens)
    plot_dists_prior(S93.dists_prior, plot_dir, S93.Np)

    # plot_obs_examples(HMM,xps,dir)
    
    print('Call launch_experiments...')
    save_as = launch_experiments(HMM,xps)
    
    print('Call load_xps...')
    xps = xpList(load_xps(save_as))
    print_summary(xps,S93.Np,S93.dists_prior)

    # Associate each control variable with a "coordinate"
    # xp_dict = xpSpace.from_list(xps)

    # plot prior+post distributions
    plot_dists_xps_onerow(xps, S93.dists_prior, plot_dir, S93.Np)
    