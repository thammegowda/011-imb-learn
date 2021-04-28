#!/usr/bin/env bash


#SBATCH --partition=isi
#SBATCH --mem=12G
#SBATCH --time=0-23:58:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:1
#SBATCH --output=R-%x.out.%j
#SBATCH --error=R-%x.err.%j
#SBATCH --export=NONE

# Pipeline script for MT
#
# Author = Thamme Gowda (tg@isi.edu)
# Date = April 3, 2019
# Last revised: April 20, 2021

#SCRIPTS_DIR=$(dirname "${BASH_SOURCE[0]}")  # get the directory name
#RTG_PATH=$(realpath "${SCRIPTS_DIR}/..")

# If using compute grid, and dont rely on this relative path resolution, set the RTG_PATH here
#RTG_PATH=/full/path/to/rtg-master
SRC_PATH=../


export N_CPUS=8
export OMP_NUM_THREADS=$N_CPUS
export MKL_NUM_THREADS=$N_CPUS

OUT=
CONF_PATH=

#defaults
#CONDA_ENV=rtg     # empty means don't activate environment
CONDA_ENV=rtg
source ~/.bashrc

usage() {
    echo "Usage: $0 -d <exp/dir>
    [-e conda_env  default:$CONDA_ENV (empty string disables activation)] " 1>&2;
    exit 1;
}


while getopts ":fd:c:e:p:" o; do
    case "${o}" in
        d) OUT=${OPTARG} ;;
        e) CONDA_ENV=${OPTARG} ;;
        *) usage ;;
    esac
done


[[ -n $OUT ]] || usage   # show usage and exit


echo "Output dir = $OUT"
[[ -d $OUT ]] || mkdir -p $OUT
OUT=`realpath $OUT`

if [[ -n ${CONDA_ENV} ]]; then
    echo "Activating environment $CONDA_ENV"
    conda activate ${CONDA_ENV} || { echo "Unable to activate $CONDA_ENV" ; exit 3; }
    pip freeze > $OUT/requirements.txt
fi

if [[ ! -f $OUT/src.zip ]]; then
    [[ -f $SRC_PATH/imgcls/__init__.py ]] || { echo "Error: SRC_PATH=$SRC_PATH is not valid"; exit 2; }
    echo "Zipping source code to $OUT/src.zip"
    OLD_DIR=$PWD
    cd ${SRC_PATH}
    zip -r $OUT/src.zip imgcls -x "*__pycache__*"
    git rev-parse HEAD > $OUT/githead   # git commit message
    cd $OLD_DIR
fi


export PYTHONPATH=$OUT/src.zip
#export PYTHONPATH=$SRC_PATH
# copy this script for reproducibility
cp "${BASH_SOURCE[0]}"  $OUT/job.sh.bak
echo  "`date`: Starting pipeline... $OUT"


cmd="python -m imgcls.train $OUT"
echo "command::: $cmd"
if eval ${cmd}; then
    echo "`date` :: Done"
else
    echo "Error: exit status=$?"
fi
