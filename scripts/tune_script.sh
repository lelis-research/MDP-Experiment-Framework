#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=0-10:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=<rrg-lelis>
#SBATCH --mail-user=aghakasi@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

source ~/MDP-Experiment-Framework/venvs/rl_v1/bin/activate
python tune_hp.py --env MiniGrid-ChainEnv-v0 --agent A2C_v1 --num_episodes 200 