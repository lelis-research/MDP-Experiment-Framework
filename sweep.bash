#!/bin/bash

# Arrays for the parameters to sweep
num_options_list=(1 5 10 20)
last_percentage_list=(5 50 100)
run_ind_list=($(seq 1 10))  # Generates values from 1 to 10

# Fixed parameters
masked_layers=("1" "input")      # Pass as a single element list
search_budget=100

for num_options in "${num_options_list[@]}"; do
    for last_percentage in "${last_percentage_list[@]}"; do
        # The innermost loop: run_ind is executed in parallel
        for run_ind in "${run_ind_list[@]}"; do
            (
                echo "Running learn_options.py with:"
                echo "  num_options = $num_options"
                echo "  masked_layers = [${masked_layers[@]}]"
                echo "  last_percentage_of_trajectories = $last_percentage"
                echo "  run_ind = $run_ind"
                echo "  search_budget = $search_budget"
                
                python learn_options.py \
                    --num_options "$num_options" \
                    --masked_layers "${masked_layers[@]}" \
                    --last_percentage_of_trajectories "$last_percentage" \
                    --run_ind "$run_ind" \
                    --search_budget "$search_budget"
                
                echo "------------------------------------------"
            ) &
        done
        # Wait for all background processes (all run_ind iterations) to finish
        wait
    done
done