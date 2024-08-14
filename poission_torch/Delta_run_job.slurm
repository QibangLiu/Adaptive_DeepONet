#!/bin/bash
#SBATCH -J AI
#SBATCH --output=./slurm_output/runjobs_%j.log
#SBATCH --account=bblv-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=01:30:00      # hh:mm:ss for the jo
#SBATCH --mem=199g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
module load anaconda3_gpu/23.3.1
which python
# # srun python pi-deeponet_2D_mydata.py
# # which python
dN='400'
start='0'
end='25'
k='0'
c='1'
case='2'
srun python3 ddm-deeponet_adaptive.py  $dN  $start $end $k $c $case