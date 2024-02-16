#!/bin/bash

### on the euler cluster at eth
### run as > bash this_file.sh

### called from within a DAPPER job, this is not a slurm file!

### directories
# ENS_DIR='./'
PYCLES_DIR='/cluster/work/climate/dgrund/git/pressel/pycles/'
PYCLES_DIR_ADD='/cluster/work/climate/dgrund/git/dana-grund/doctorate_code/pycles/'
NAMELIST_GENERATOR=${PYCLES_DIR_ADD}generate_namelist_ensemble.py
ENSEMBLE_RUNNER=${PYCLES_DIR_ADD}run_ensemble.py
# PLOTTER=${PYCLES_DIR_ADD}Straka_plots.py # could be used when configuring command-line usage

### activate environment
source /cluster/work/climate/dgrund/git/dana-grund/doctorate_code/euler/euler_setup_template.sh

### make target direction
# mkdir $ENS_DIR
# cd $ENS_DIR
source parameters.sh

# ### remove old results
# cd $ENS_DIR
# rm -rf O*
# echo 'Removed results in '$ENS_DIR

# ### recompile pycles
# cd $PYCLES_DIR
# CC=mpicc python ${PYCLES_DIR}setup.py build_ext --inplace

### generate in_files
NPROCX=1 # see performance analysis in 23-09-25_SP_sampling/eval_nproc.ipynb
python $NAMELIST_GENERATOR StableBubble -p $ENS_DIR -np $NPROCX -v $V -d $D -r $R -T $T # -t 

### define inifle to run
# cd $ENS_DIR
# IN_FILE='StableBubble_default.in'
# IN_FILE='StableBubble_v75.000d75.000a10.0xr2.0zr2.0zc3.0.in'
# IN_FILE='StableBubble_v75.000d75.000a15.0xr4.0zr2.0zc3.0.in'
# IN_FILE='StableBubble_v75.000d75.000a20.0xr6.0zr2.0zc3.0.in'

### run one infile
# echo 'Simulating...'
python $ENSEMBLE_RUNNER -p $ENS_DIR
# python ${PYCLES_DIR}main.py ${ENS_DIR}${IN_FILE} # if infile is known
# echo '...simulating done.'

# ### plot
# echo 'Plotting...'
# python $PLOTTER -p $ENS_DIR -v 'temperature' -i $IN_FILE
# echo '...plotting done.'

