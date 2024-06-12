#!/bin/bash

### on the euler cluster at eth:
### run as > bash this_file.sh
### this is not a slurm submit file

### directories
PYCLES_DIR='/cluster/work/climate/dgrund/git/pressel/pycles/'
PYCLES_DIR_ADD='/cluster/work/climate/dgrund/git/dana-grund/doctorate_code/pycles/'
NAMELIST_GENERATOR=${PYCLES_DIR_ADD}generate_namelist_ensemble.py
ENSEMBLE_RUNNER=${PYCLES_DIR_ADD}run_ensemble.py

### input: ensemble directory
ENS_DIR=${1:-'./'}
echo Found ENS_DIR = $ENS_DIR

### activate environment
source /cluster/work/climate/dgrund/git/dana-grund/doctorate_code/euler/euler_setup_template.sh

### input saved by DAPPER
source parameters.sh

### default state writing frequency
F_STATE=300
if [ $T -lt 300 ]; then
    F_STATE=$T
fi

### generate input file
NPROCX=8
python $NAMELIST_GENERATOR StableBubble \
    -p $ENS_DIR -np $NPROCX \
    -v $V -d $D \
    -r $R -T $T \
    --f_state $F_STATE --f_stats 10 --f_cstats 10 \
    --meas_locs 10000 0 500 \
    --meas_locs 10000 0 2000 \
    --meas_locs 20000 0 500 \
    --meas_locs 20000 0 2000
    
### submit in a job
ID=$(sbatch \
    -J S  \
    -n $NPROCX  \
    --parsable  \
    --mem-per-cpu=1G  \
    --time=01:00:00  \
    --output=slurm.out  \
    --error=slurm.err \
    --wrap="python $ENSEMBLE_RUNNER -p $ENS_DIR > ${ENS_DIR}pycles.out" \
    )
echo Submitted ID=$ID
