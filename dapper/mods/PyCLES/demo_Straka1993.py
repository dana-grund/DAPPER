print('Importing...')
from mpl_tools import is_notebook_or_qt as nb
import argparse
# print('da')
import dapper.da_methods as da
# print('Straka')
from dapper.mods.PyCLES.Straka1993 import create_HMM, set_X0_and_simulate, Np, dists_prior, plot_xt # incl HMM
# print('PyCLES')
from dapper.mods.PyCLES import * # load_xps, print_summary, dists_single_xp, plot_dists
print('...importing done.')

def set_up_experiments(N_ens):
    xps = dpr.xpList()
    # xps += da.PartFilt(N=1000, reg=1)
    ''' On the ensemble size
    - N>= 2 as we need N-1 for variances
    - One model eval more used for data generation in set_X0_and_simulate()
    '''
    xps += da.EnKF('Sqrt', N_ens, infl=1.04)
    # xps += da.EnKF_N(N_ens, xN=2) # xN for hyperprior coeffs
    xps += da.iEnKS("Sqrt", N_ens, Lag=1, xN=2, nIter=5, wtol=1e-5)
    return xps

def launch_experiments(HMM,xps,N_ens):
    # HMM.mp = N_ens # one process per member # does not work here! set manually.
    # XXX caution with large ensembles!
    scriptname = HMM.name if nb else __file__
    save_as = xps.launch(
        HMM, scriptname, setup=set_X0_and_simulate,
        mp=False,           # Multiprocessing
        fail_gently=False,  # Facilitate debugging
        liveplots=False,    # NB: Turn off if running iEnKS
        free=False,         # Don't delete time series (enables replay)
    )
    return save_as

def plot_obs_examples(HMM,xps,plot_dir):
    HMM, xx, yy = set_X0_and_simulate(HMM,xps[0])
    plot_xt(yy[0],plot_dir)



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(prog='')
    parser.add_argument('-p','--path', type=str, default='./',
                        help='Where to store figs and data in folder p/data/')
    parser.add_argument('-o','--obs_type', type=str, default='full',
                        help='Type ob obervations, see Straka1993.py')
    parser.add_argument('-N','--N_ens', type=int, default=20,
                        help='Number of ensemble members (>=2)')
    args = parser.parse_args()
    
    plot_dir = args.path # '/cluster/work/climate/dgrund/working_dir/24-01-07_dapper_straka/T900_fullobs_obsstd0.1/'
    data_dir = plot_dir + 'data/'
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    N_ens = args.N_ens
    obs_type = args.obs_type

    HMM = create_HMM(mp=N_ens, data_dir=data_dir, obs_type=obs_type)

    xps = set_up_experiments(N_ens)
    plot_dists_prior(dists_prior, plot_dir, Np)

    # plot_obs_examples(HMM,xps,plot_dir)
    
    save_as = launch_experiments(HMM,xps,N_ens)
    
    xps = load_xps(save_as)
    print_summary(xps,Np,dists_prior)

    # Associate each control variable with a "coordinate"
    xp_dict = dpr.xpSpace.from_list(xps)

    # plot prior+post distributions
    # plot_dists_xps(xps, dists_prior, plot_dir, Np)
    plot_dists_xps_onerow(xps, dists_prior, plot_dir, Np)
    
    # plot_target_function()