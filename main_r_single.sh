#!/bin/bash
#
#SBATCH -p gpu                    # partition (queue)
#SBATCH --qos=valhala
#SBATCH --nodes=1                 # number of nodes
#SBATCH --ntasks-per-node=2       # number of cores
#SBATCH --mem=10G                 # memory pool for all cores
#SBATCH -t 1-00:00                # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out        # STDOUT
#SBATCH -e slurm.%N.%j.err        # STDERR
# source ../rnn_generator_env/bin/activate
optimization_steps=6
for ((i = 1; i <= $optimization_steps; i++)); do
    echo "Optimization step: $i / $optimization_steps"

    python gridsearch.py

    # srun python optimize_single.py --optimization_steps 3 --num_nodes 1 --devices [0,1] --accelerator gpu

    # python update.py
done
