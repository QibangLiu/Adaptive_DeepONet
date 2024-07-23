# # ./bash_jupyter.sh > ./slurm_output/jupyter_bash_output.log 2>&1 &
MYPORT=$(($(($RANDOM % 10000))+49152)); echo $MYPORT
srun --account=bbpq-delta-gpu --ntasks-per-node=1 \ 
--partition=gpuA100x4-interactive --time=00:40:00 \
--mem=199g --gpus-per-node=1 --gpus-per-task=1 \
jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0