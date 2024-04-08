#!/bin/bash
#
#SBATCH -p gpu                    # partition (queue)
#SBATCH --qos=valhala
#SBATCH --nodes=1                 # number of nodes
#SBATCH --ntasks-per-node=1       # number of cores
#SBATCH --mem=10G                 # memory pool for all cores
#SBATCH -t 1-00:00                # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out        # STDOUT
#SBATCH -e slurm.%N.%j.err        # STDERR

source ../rnn_generator_env/bin/activate

srun python optimize_single.py --optimization_steps 3 --num_nodes 1 --num_devices 2 --accelerator gpu