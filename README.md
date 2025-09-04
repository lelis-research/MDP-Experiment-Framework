# Installing Requirements
- Python 3.11.5
- Dependencies from `Requirements/requirements.txt`

# Usage

  ## Training Agents

    You can train agents either directly with `train.py` or by submitting jobs to a SLURM cluster using the provided `train_script.sh`.

    ### 1. Direct Training (Local or Single Node)

    Run the following command (adjust parameters as needed):

    ```bash
    python train.py \\
      --agent A2C \\
      --env MiniGrid-SimpleCrossingS9N1-v0 \\
      --seed 1234 \\
      --num_runs 1 \\
      --total_steps 500000 \\
      --episode_max_steps 300 \\
      --num_envs 1 \\
      --name_tag "example" \\
      --info '{"gamma":0.99,"lamda":0.95,"actor_network":"fc_network_relu"}' \\
      --env_wrapping '["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]' \\
      --wrapping_params '[{"agent_view_size":9},{},{"seed":10000},{"num_distractors":25,"seed":100}]'
    ```

    #### Main Arguments
    - **config**: Config file to load agent definitions (default: `config_agents_base`).
    - **agent**: Type of agent (see `Configs/config_agents_base.py` for available agents).
    - **env**: Environment name (must be in `ENV_LST`).
    - **seed**: Random seed for reproducibility.
    - **num_runs**: Number of independent runs.
    - **num_episodes**: Episodes per run (**either this OR total_steps**).
    - **total_steps**: Total number of environment steps (**either this OR num_episodes**).
    - **episode_max_steps**: Maximum steps per episode.
    - **num_envs**: Number of parallel environments.
    - **render_mode**: Rendering mode (`None`, `human`, `rgb_array_list`).  
      *Note: `rgb_array_list` will store raw frames in the experiment directory, which takes significant space.*
    - **store_transitions**: Save all transitions (`--store_transitions` flag). By default off.
    - **checkpoint_freq**: Frequency of saving checkpoints. `0` means only the last checkpoint is saved.
    - **name_tag**: A tag for the experiment name.
    - **num_workers**: Number of parallel workers for rollouts.
    - **info**: JSON dictionary for agent hyperparameters (e.g., learning rate, entropy coefficient, etc.).
    - **env_params**: JSON dictionary of environment parameters (e.g., `{"continuing_task":false}`).
    - **env_wrapping**: JSON list of wrappers to apply (e.g., `["NormalizeObs","ClipReward"]`).
    - **wrapping_params**: JSON list of dictionaries, one per wrapper, specifying wrapper-specific arguments.

    All results (trained models, logs, and plots) are saved under:
    ```
    Runs/Train/<env>/<wrappers>/<agent>/<name_tag>_seed[<seed>]/
    ```

    To view logs in TensorBoard:
    ```bash
    tensorboard --logdir=Runs/Train/
    ```

    ---

    ### 2. Training on SLURM (Cluster)

      Use the provided script:

      ```bash
      sbatch train_script.sh
      ```

      The SLURM script:
        - Uses job arrays (`--array=0-50`) to run multiple seeds/configs.
        - Loads required modules (`mujoco`) and activates your Python environment.
        - Passes environment wrappers, seeds, and hyperparameters to `train.py`.
        - Collects job stats at the end with `seff` or `sacct`.

      Modify the variables in **train_script.sh** to set:
        - Agent (`AGENT`)
        - Environment (`ENV`)
        - Wrappers (`ENV_WRAPPING` and `WRAPPING_PARAMS`)
        - Seed (`SEED=$SLURM_ARRAY_TASK_ID`)
        - Steps, episodes, workers, etc.

      This makes it easy to launch large sweeps on Compute Canada clusters.

  

## Tuning Hyper-Params

  To tune hyper-parameters using Optuna, run the tune_hp.py script. For example:
  ```
  python tune_hp.py --agent PPO --env MiniGrid-Empty-5x5-v0 --seed 1 --num_configs 300 --num_runs 2 --num_episodes 200 --episode_max_steps 200 --num_envs 1 --metric_ratio 0.5
  ```
  - You can modify the search ranges for each hyper-parameter in `tune_hp.py`.
  
  > Similarly You can modify the **config.py** to change default hyper parameters (search starting point) or change the wrappers on the environment.

# Defining New Agents

  To add a new agent to the framework, follow these simple steps:
  
  1. **Create your agent file:**  
    Place your new agent implementation (e.g., `MyAgent.py`) in an appropriate subfolder of `Agents/`.
  
  2. **Implement the agent class:**  
    Extend `BaseAgent` (or a suitable base class) and implement the required methods such as `act()`, `update()`, and `reset()`.
  
  3. **Register your agent:**
    Update the `__init__.py` of the new agent directory.
    
  4. **Set default parameters:**
    Update the agent dictionary (`config.py`) by adding a new entry in `AGENT_DICT`. For example:
    
    ```python
      "MyAgent": lambda env: MyAgent(
          get_env_action_space(env),
          get_env_observation_space(env),
          HyperParameters(your_params_here),
          get_num_envs(env),
          YourFeatureExtractor,
      )
    ```
    
# Defining New Environments

  To add a new environment to the framework, follow these steps:
  
  1. **Register the Environment:**
     Ensure your environment is registered with Gymnasium.
  
  2. **Implement the Wrappers:**
     Implement the wrappers using the Gymnasium observation, reward, action wrappers.
     
  3. **Implement Environment Creation Functions:**  
    In the appropriate folder (e.g., `Environments/NewEnv/``), create `get_single_env()` and `get_parallel_env()` to properly instantiate your environment and apply any necessary wrappers.
  
  4. **Update the Global Factory:**  
    In `Environments/__init__.py`, add your new environment to the combined environment list (`ENV_LST`) and ensure the `get_env()` function can instantiate it with the correct parameters and wrappers.
  
  After these steps, your environment will be available for training and evaluation through the framework.

