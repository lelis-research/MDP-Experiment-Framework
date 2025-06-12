#!/bin/bash
# Backup the original config.py file
cp config.py config.py.bak

# Loop over the combinations:
# R: from 1 to 10
# T: 5, 50, 100
# N: 1, 5, 10, 20
for R in {1..10}; do
    for T in 5 50 100; do
        for N in 1 5 10 20; do
            echo "Running train.py with R=${R}, T=${T}, N=${N}"
            
            # Update the initial_options file name in config.py
            # Using extended regex (-E) and the BSD sed syntax for in-place editing (-i '')
            sed -i '' -E "s/R[0-9]+_T[0-9]+_N[0-9]+/R${R}_T${T}_N${N}/" config.py
            
            # Run train.py with the fixed parameters and agent MaskedDQN.
            python train.py --agent MaskedDQN --env MiniGrid-ChainEnv-v1 --num_runs 5 --num_episodes 300 --name_tag L[1]_R${R}_T${T}_N${N}
            
            echo "Finished run for R=${R}, T=${T}, N=${N}"
            echo "---------------------------------------------"
        done
    done
done

# Restore the original config.py file
mv config.py.bak config.py