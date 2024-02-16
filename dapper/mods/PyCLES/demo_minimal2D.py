import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_tools import is_notebook_or_qt as nb
import scipy.stats as stats

import dapper as dpr
import dapper.da_methods as da
import dapper.mods as modelling
import dapper.tools.liveplotting as LP
from dapper.mods.PyCLES.minimal2D import * # incl HMM

print('PyCLES')
from dapper.mods.PyCLES import load_xps, print_summary, dists_single_xp, plot_dists

def set_up_experiments():
    xps = dpr.xpList()
    # xps += da.PartFilt(N=1000, reg=1)
    for N in [20]: # >= 2 as we need N-1 for variances
        xps += da.EnKF('Sqrt', N, infl=1.04)
        xps += da.EnKF_N(N, xN=2)
        xps += da.iEnKS("Sqrt", N, Lag=1, xN=2, nIter=10, wtol=1e-5)
    return xps

def plot_obs_examples(HMM,xps,plot_dir):
    HMM, xx, yy = set_X0_and_simulate(HMM,xps[0])
    print('xx',xx)
    print('yy',yy)


    Mperdim = int(np.sqrt(Nx))
    a = np.arange(Mperdim)
    term1 = 0.01*a**2
    term2 = np.sin(a)
    term3 = np.cos(a)

    v_field = np.outer(term2,term3)
    d_field = np.outer(term3,term1)

    v=d=1
    x_t = 0.5*v**2*v_field + 0.5*d**3*d_field

    plt.imshow(v_field)
    plt.colorbar()
    plt.title('v')
    plt.savefig(plot_dir+'fig-min2D_v_field.png')
    plt.show()

    plt.imshow(d_field)
    plt.colorbar()
    plt.title('d')
    plt.savefig(plot_dir+'fig-min2D_d_field.png')
    plt.show()

    plt.imshow(x_t)
    plt.colorbar()
    plt.title('data')
    plt.savefig(plot_dir+'fig-min2D_sampleData.png')
    plt.show()

    ## if identity data
    # plt.imshow(yy[0].reshape((Mperdim,Mperdim)))
    # plt.colorbar()
    # plt.title('data yy')
    # plt.savefig(plot_dir+'fig-min2D_data')
    # plt.show()

def launch_experiments(HMM,xps):
    scriptname = HMM.name if nb else __file__
    save_as = xps.launch(
        HMM, scriptname, setup=set_X0_and_simulate,
        mp=False,           # Multiprocessing
        fail_gently=False,  # Facilitate debugging
        liveplots=False,    # NB: Turn off if running iEnKS
        free=False,         # Don't delete time series (enables replay)
    )
    return save_as



# one scalar measurement plotted --> reaction depends on location!
def plot_target_function():
    space = np.arange(-3,3,0.5)
    Ns = len(space)
    Mperdim = int(np.sqrt(Nx))
    result = []
    for v,d in itertools.product(space,space):
        x = np.ones((Mperdim,Mperdim))

        # -- Obs = partial_Id_Obs size 3
        obs = call_dummy(x,[v,d],0,0) 
        result.append([v,d, obs[0], obs[1], obs[2]])

        # # -- Obs = Id_Obs
        # observable = call_dummy(x,[v,d],0,0)[13] 
        # result.append([v,d, observable])

    result = np.array(result).T

    fig,axs = plt.subplots(3,2)
    for i in range(3):
        axs[i,0].scatter(result[0], result[2+i])
        axs[i,0].set_xlabel('parameter 0')

        axs[i,1].scatter(result[1], result[2+i])
        axs[i,1].set_xlabel('parameter 1')

    plt.savefig(plot_dir+'fig-min2D_targetSamples.png')

if __name__=='__main__':
    plot_dir = 'figs-Minimal2D/'

    xps = set_up_experiments()
    plot_obs_examples(HMM,xps,plot_dir)
    save_as = launch_experiments(HMM,xps)
    
    xps = load_xps(save_as)
    print_summary(xps)

    # Associate each control variable with a "coordinate"
    xp_dict = dpr.xpSpace.from_list(xps)

    # plot single experiment
    dists = dists_single_xp(xps[0])
    plot_dists(dists,plot_dir)
    
    # plot_target_function()