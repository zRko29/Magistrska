#!/bin/bash

max_iter=2
sleep_time=3
train_file="trainer.py"
update_file="update_params.py"

for i in $(seq $max_iter); do
    echo
    echo "Starting iteration $i"
    (
        (
            python $train_file
            echo "Finished '$train_file' process 1 in iteration $i"
        ) &
        (
            sleep $sleep_time
            python $train_file
            echo "Finished '$train_file' process 2 in iteration $i"
        ) &
        (
            sleep $((sleep_time * 2))
            python $train_file
            echo "Finished '$train_file' process 3 in iteration $i"
        ) &
        (
            sleep $((sleep_time * 3))
            python $train_file
            echo "Finished '$train_file' process 4 in iteration $i"
        )
    )
    python $update_file
    echo "'$update_file' completed in iteration $i"
    echo "---------------------------------"
    echo
done
