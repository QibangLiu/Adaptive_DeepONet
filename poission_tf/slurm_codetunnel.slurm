#!/bin/bash
#SBATCH -J vscode_tunnel
#SBATCH --output=./slurm_output/vscode_tunnel%j.log
#SBATCH --account=bblv-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=00:10:00      # hh:mm:ss for the jo
#SBATCH --mem=200g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
export PATH="$PATH:/u/qibang/software"
#MYPORT=$(($(($RANDOM % 10000))+49152)); echo $MYPORT --port=8891
##export LD_LIBRARY_PATH=/sw/external/python/anaconda3-2023.03_cuda/lib:$LD_LIBRARY_PATH
##srun jupyter-lab --no-browser  --ip=0.0.0.0 #srun
code tunnel --accept-server-license-terms
