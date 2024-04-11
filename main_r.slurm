#!/bin/bash
#
#SBATCH --partition=gpu                 # partition (queue)
#SBATCH --qos=valhala
#SBATCH --nodelist=valhala-gpu1         # node list
#SBATCH --ntasks-per-node=2             # number of cores
#SBATCH --mem=24G                       # memory pool
#SBATCH --time=1-00:00                  # time (D-HH:MM)
#SBATCH --output=slurm.%N.%j.out        # STDOUT
#SBATCH --error=slurm.%N.%j.err         # STDERR

export NCCL_P2P_DISABLE=1

source ~/.bashrc

conda activate rnn_env

python gridsearch.py

srun python trainer.py --devices 2 --train_size 1.0 --num_epochs 4000

python update.py --min_good_samples 3 --max_good_loss 5e-6