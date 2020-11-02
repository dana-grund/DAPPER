"""Illustrate usage of DAPPER to benchmark multiple DA methods."""

from dapper import *
set_seed(3000)

##############################
# DA method configurations
##############################
xps = xpList()

from dapper.mods.Lorenz63.sakov2012 import HMM   # Expected rmse.a:
xps += Climatology()                                     # 7.6
xps += OptInterp()                                       # 1.25
xps += Var3D(xB=0.1)                                     # 1.03 
xps += ExtKF(infl=90)                                    # 0.87
xps += EnKF('Sqrt',    N=3 ,  infl=1.30)                 # 0.82
xps += EnKF('Sqrt',    N=10,  infl=1.02,rot=True)        # 0.63
xps += EnKF('PertObs', N=500, infl=0.95,rot=False)       # 0.56
xps += EnKF_N(         N=10,            rot=True)        # 0.54
xps += iEnKS('Sqrt',   N=10,  infl=1.02,rot=True)        # 0.31
xps += PartFilt(       N=100 ,reg=2.4  ,NER=0.3)         # 0.38
xps += PartFilt(       N=800 ,reg=0.9  ,NER=0.2)         # 0.28
# xps += PartFilt(     N=4000,reg=0.7  ,NER=0.05)        # 0.27
# xps += PFxN(xN=1000, N=30  ,Qs=2     ,NER=0.2)         # 0.56

# from dapper.mods.Lorenz96.sakov2008 import HMM  # Expected rmse.a:
# xps += Climatology()                                     # 3.6
# xps += OptInterp()                                       # 0.95
# xps += Var3D(xB=0.02)                                    # 0.41 
# xps += ExtKF(infl=6)                                     # 0.24
# xps += EnKF('PertObs'        ,N=40,infl=1.06)            # 0.22
# xps += EnKF('Sqrt'           ,N=28,infl=1.02,rot=True)   # 0.18
# 
# xps += EnKF_N(N=24,rot=True)                             # 0.21
# xps += EnKF_N(N=24,rot=True,xN=2)                        # 0.18
# xps += iEnKS('Sqrt',N=40,infl=1.01,rot=True)             # 0.17
# 
# xps += LETKF(         N=7,rot=True,infl=1.04,loc_rad=4)  # 0.22
# xps += SL_EAKF(       N=7,rot=True,infl=1.07,loc_rad=6)  # 0.23

# Other models (suitable xp's listed in HMM files):
# from dapper.mods.LA           .evensen2009 import HMM
# from dapper.mods.KS           .bocquet2019 import HMM
# from dapper.mods.LotkaVolterra.settings101 import HMM

##############################
# Run experiment
##############################
# Adjust experiment duration
HMM.t.BurnIn = 2
HMM.t.T = 50

# Assimilate (for each xp in xps)
xps.launch(HMM,liveplots=False)

# Print results
print(xps.tabulate_avrgs())
