current_time=$(date +"%Y-%m-%d-%H:%M:%S")
filename="./slurm_output/jupyter_${current_time}.log"
./bash_jupyter.sh > $filename 2>&1 &
