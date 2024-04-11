#!/bin/bash
#
#SBATCH --partition=gpu                        # partition (queue)
#SBATCH --qos=valhala
#SBATCH --nodes=2                              # number of nodes
#SBATCH --ntasks-per-node=2                    # number of cores
#SBATCH --mem=0                                # memory pool for all cores
#SBATCH --time=1-00:00                         # time (D-HH:MM)
#SBATCH --output=slurm.%N.%j.out               # STDOUT
#SBATCH --error=slurm.%N.%j.err                # STDERR

source ~/.bashrc

conda activate rnn_env

optimization_steps=30

for i in $(seq $optimization_steps)
do
    echo
    echo "-----------------------------"
    echo
    echo "Optimization step: $i / $optimization_steps"
    echo
    
    python gridsearch.py
    
    srun python trainer.py --nodes 2 --devices 2 --accelerator gpu --strategy ddp --train_size 1.0 --num_epochs 4000
    
    python update.py --min_good_samples 3 --max_good_loss 5e-6 --check_every_n_steps 3 --current_step $i
    
done
