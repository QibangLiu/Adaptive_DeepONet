#!/bin/bash
#SBATCH -J AI
#SBATCH --output=./slurm_output/AdaptiveLearning%j.log
#SBATCH --account=bcxr-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4-interactive     # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8-interactive
#SBATCH --time=00:20:00      # hh:mm:ss for the jo
#SBATCH --mem=200g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
module load anaconda3_gpu/23.3.1
export LD_LIBRARY_PATH=/sw/external/python/anaconda3-2023.03_cuda/lib:$LD_LIBRARY_PATH
which python
dN='50'
start='0'
end='2'
k='0'
c='0'
case='2'

srun python3 ddm-deeponet_adaptive_tf.py $dN  $start $end $k $c $case