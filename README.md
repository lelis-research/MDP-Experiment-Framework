# Installing Requirements
- Python 3.10
- Dependencies from `Requirements/requirements.txt`

# Usage

## Training Agent

  Run the following command (adjust parameters as needed):
  ```
  python train.py --agent PPO --env MiniGrid-Empty-5x5-v0 --seed 123123 --num_runs 3 --num_episodes 200 --episode_max_steps 500 --num_envs 1 --render_mode None
  ```
  - agent: Type of agent (list of all agents in config.py).
  - env: Environment name.
  - seed: Random seed.
  - num_runs: Number of independent runs.
  - num_episodes: Episodes per run.
  - episode_max_steps: Maximum steps per episode.
  - num_envs: Number of parallel environments.
  - render_mode: Rendering mode (None, human, rgb_array_list): rgb_array_list will store the frames in experiment directory which takes a lot of space
  - store_transitions: Boolean indication to save all the transition in the result directory which takes a lot of space (by default false)
  - checkpoint_freq: Frequency of saving checkpoints (by default only the last policy will be saved)
  
  > You can modify the **config.py** to set specific hyper parameters or change the wrappers on the environment.
  
  Trained models, metrics, and logs are saved under Runs/Train/. To view logs with TensorBoard, run:
  ```
  tensorboard --logdir=Runs/Train/
  ```

  

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

