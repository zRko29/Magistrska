#!/bin/bash
#
#SBATCH -p gpu                    # partition (queue)
#SBATCH --qos=valhala
#SBATCH --nodes=2                 # number of nodes
#SBATCH --ntasks-per-node=4       # number of cores
#SBATCH --mem=10G                 # memory pool for all cores
#SBATCH -t 1-00:00                # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out        # STDOUT
#SBATCH -e slurm.%N.%j.err        # STDERR

source rnn_generator_env/bin/activate

cd rnn_autoregression_model

srun python optimize.py --optimization_steps 2 --models_per_step 1 --num_nodes 2 --strategy ddp --num_devices 4 --accelerator gpu