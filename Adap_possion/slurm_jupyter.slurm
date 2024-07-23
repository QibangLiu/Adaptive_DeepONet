#!/bin/bash
#SBATCH -J Jupyter
#SBATCH --output=./slurm_output/jupyterServer_A100_%j.log
#SBATCH --account=bbpq-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA100x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=01:00:00      # hh:mm:ss for the jo
#SBATCH --mem=199g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
###module load anaconda3_gpu/23.3.1
MYPORT=$(($(($RANDOM % 10000))+49152)); echo $MYPORT
srun jupyter lab --no-browser --port=8891 --ip=0.0.0.0
###srun jupyter-notebook --no-browser --port=8991 --ip=0.0.0.0 ## rm srun

# # # ##SBATCH --gpu-bind=verbose,per_task:1
# # # ###SBATCH --gpu-bind=none     # <- or closest


# # # ##SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
# # # srun jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0
# # # #srun jupyter lab --no-browser --port=8991 --ip=0.0.0.0

# ## ssh -l qibang -L 127.0.0.1:56673:gpub002.delta.internal.ncsa.edu:56673 dt-login.delta.ncsa.illinois.edu
#ssh -l qibang -L 127.0.0.1:8991:gpub001.delta.ncsa.illinois.edu:8991 dt-login.delta.ncsa.illinois.edu
# ##srun --account=bbpq-delta-gpu --ntasks-per-node=1 --partition=gpuA100x4-interactive --time=00:40:00 --mem=199g --gpus-per-node=1 --gpus-per-task=1 jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0