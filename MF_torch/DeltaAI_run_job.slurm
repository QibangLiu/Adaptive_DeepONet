#!/bin/bash
#SBATCH -J AI
#SBATCH --output=./slurm_output/DeltaAIjobs_%j.log
#SBATCH --account=bbkg-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ghx4     # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=02:30:00      # hh:mm:ss for the jo
#SBATCH --mem=199g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
module load python/miniforge3_pytorch/2.5.0
which python
# # srun python pi-deeponet_2D_mydata.py
# # which python
dN='200'
start='0'
end='25'
k='4.0'
c='0'
case='0'
srun python3 ddm-deeponet_adaptive.py  $dN  $start $end $k $c $case