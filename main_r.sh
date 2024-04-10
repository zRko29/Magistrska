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

cd rnn_autoregression_model/

export NCCL_DEBUG=WARN # WARN

optimization_steps=20

for i in $(seq $optimization_steps)
do
    echo
    echo "-----------------------------"
    echo
    echo "Optimization step: $i / $optimization_steps"
    echo
    
    # update current_params.yaml
    python gridsearch.py
    
    # train new model
    srun python trainer.py --num_nodes 2 --devices 2 --strategy ddp --accelerator gpu --train_size 0.8 --num_epochs 4000
    
    # update gridsearch intervals
    python update.py --min_good_samples 3 --max_good_loss 5e-6 --check_every_n_steps 3 --current_step $i
    
done
