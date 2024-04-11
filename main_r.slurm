#!/bin/bash
#
#SBATCH --partition=gpu                        # partition (queue)
#SBATCH --qos=valhala
#SBATCH --nodes=1                              # number of nodes
#SBATCH --ntasks-per-node=2                    # number of cores
#SBATCH --mem=10G                              # memory pool for all cores
#SBATCH --time=1-00:00                         # time (D-HH:MM)
#SBATCH --output=slurm.%N.%j.out               # STDOUT
#SBATCH --error=slurm.%N.%j.err                # STDERR
#SBATCH --mail-type=END                        # Send email at job completion
#SBATCH --mail-user=grandovecuros@gmail.com    # Email address for notifications

source ~/.bashrc

conda activate rnn_env

optimization_steps=2

for i in $(seq $optimization_steps)
do
    echo
    echo "-----------------------------"
    echo
    echo "Optimization step: $i / $optimization_steps"
    echo
    
    # update config/current_params.yaml
    python gridsearch.py
    
    # train new model
    srun python trainer.py --num_nodes 1 --devices 2 --strategy ddp --accelerator gpu --train_size 0.8 --num_epochs 10
    
    # update gridsearch intervals
    python update.py --min_good_samples 3 --max_good_loss 5e-6 --check_every_n_steps 3 --current_step $i
    
done

rm config/current_params.yaml
