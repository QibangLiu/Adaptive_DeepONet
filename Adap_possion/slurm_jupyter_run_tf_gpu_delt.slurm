#!/bin/bash
#SBATCH -J Jupyter
#SBATCH --output=./slurm_output/jupyterServer_A100_%j.output
#SBATCH --account=bbpq-delta-gpu
#SBATCH --nodes=1
# # #SBATCH --ntasks-per-node=1
# # #SBATCH --partition=gpuA100x4-interactive      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
# # #BATCH --time=00:40:00      # hh:mm:ss for the jo
#SBATCH --mem=199g #199?

# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
# # ##SBATCH --gpu-bind=verbose,per_task:1
# # ###SBATCH --gpu-bind=closest     # <- or closest

# # ##

# # ##SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS







# # #eval "$(conda shell.bash hook)"
# # #conda deactivate
# # #conda activate base

# # #module reset
# # #module load gcc anaconda3_gpu
# # #module load anaconda3_gpu/22.10.0
# # module load anaconda3_gpu/23.3.1
# # which python3
# # ###pip show umap-learn

# # MYPORT=$(($(($RANDOM % 10000))+49152));

# # srun jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0
# # #srun jupyter lab --no-browser --port=8991 --ip=0.0.0.0

#srun python3 ./post_process/testDataDistrib.py
## ssh -l qibang -L 127.0.0.1:8991:gpua001.delta.internal.ncsa.edu:8991 dt-login.delta.ncsa.illinois.edu
MYPORT=$(($(($RANDOM % 10000))+49152)); echo $MYPORT
srun --ntasks-per-node=1 --partition=gpuA100x4-interactive --time=00:40:00 jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0
##srun --account=bbpq-delta-gpu --ntasks-per-node=1 --partition=gpuA100x4-interactive --time=00:40:00 --mem=199g --gpus-per-node=1 --gpus-per-task=1 jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0