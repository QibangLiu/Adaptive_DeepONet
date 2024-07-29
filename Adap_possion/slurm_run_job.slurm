#!/bin/bash
#SBATCH -J AI
#SBATCH --output=./slurm_output/AdaptiveLearning_A100_%j.log
#SBATCH --account=bblv-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=00:5:00      # hh:mm:ss for the jo
#SBATCH --mem=199g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
module load anaconda3_gpu/23.3.1
##export PATH="/sw/external/python/anaconda3-2023.03_cuda/bin:$PATH"
export LD_LIBRARY_PATH=/sw/external/python/anaconda3-2023.03_cuda/lib:$LD_LIBRARY_PATH
which python
method='PDF'
k='0'
c='1'
dN='400'
case='0'
start='0'
end='2'
srun python3 my_poisson_ddm-deeponet_adaptive_kmean.py $method  $dN  $start $end $k $c $case