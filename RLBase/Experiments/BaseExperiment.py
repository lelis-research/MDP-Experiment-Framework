from tqdm import tqdm
import os
import pickle
import random
import shutil
import importlib.util
import yaml
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

class BaseExperiment:
    """
    Base class for running episodes and collecting metrics.
    """
    def __init__(self, env, agent, exp_dir=None, train=True, config=None, args=None):
        """
        Args:
            env: An initialized environment.
            agent: An agent instance.
            exp_dir (str, optional): Directory to save checkpoints and metrics.
            train (bool): Whether to train the agent.
        """
         
        self._env_is_factory   = callable(env)
        self._agent_is_factory = callable(agent)
        
        # Normalize both to factories under the hood
        if self._env_is_factory:
            self._make_env = env
        else:
            self._make_env = lambda: env

        if self._agent_is_factory:
            self._make_agent = agent
        else:
            self._make_agent = lambda env: agent
            
            
        if exp_dir is not None:
            # Copy config file
            os.makedirs(exp_dir, exist_ok=True)
            self.exp_dir = exp_dir
            if config is not None:
                shutil.copy(config, os.path.join(exp_dir, "config.py"))
        
            # Save args
            file = os.path.join(self.exp_dir, "args.yaml")
            with open(file, "w") as f:
                yaml.dump(vars(args), f)

        self._dump_transitions = False
        self._checkpoint_freq = None
        self._train = train

    def _single_run_episodes(self, env, agent, num_episodes, seed, run_idx):
        """
        Run a specified number of episodes.
        
        Returns:
            list: Episode metrics for the run.
        """
        if self._train:
            agent.reset(seed)
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        for episode_idx in pbar:
            # ep_seed = episode_idx + seed # If you want each episode to have specific seeds 
            #                           (each episode is reproducible but maybe too specific)          
            observation, info = env.reset() #seed=ep_seed
            
            ep_return = 0.0
            steps = 0
            terminated = False
            truncated = False
            transitions = []
            
            while not (terminated or truncated):
                action = agent.act(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                if self._dump_transitions:
                    transitions.append((observation, action, reward, terminated, truncated))
                if self._train:
                    agent.update(next_observation, reward, terminated, truncated)
                
                ep_return += reward
                steps += 1
                observation = next_observation
            
            frames = env.render()
            metrics = {
                "ep_return": ep_return,
                "ep_length": steps,
                "frames": frames,
                # "env_seed": ep_seed,
                "transitions": transitions
            }
            
            metrics["agent_seed"] = seed
            metrics["episode_index"] = episode_idx
            all_metrics.append(metrics)
            pbar.set_postfix({
                "Return": metrics['ep_return'], 
                "Steps": metrics['ep_length'],
            })
            if self._checkpoint_freq is not None and self._checkpoint_freq != 0 and episode_idx % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Run{run_idx}_E{episode_idx}")
                agent.save(path)
        return all_metrics
    
    def _single_run_steps(self, env, agent, total_steps, seed, run_idx):
        """
        Run until we have executed 'total_steps' agent-environment interactions.
        If we reach total_steps in the middle of an episode, that episode is truncated 
        immediately.

        Args:
            total_steps (int): The total number of steps to run across one or more episodes.
            seed (int): Seed for reproducibility.
            run_idx (int): Index of the current run (used for checkpoint naming).
            
        Returns:
            list of dict: 
                A list of episode metrics. Each element is the dictionary returned 
                by self.run_episode(...), plus extra info (agent_seed, etc.).
        """
        if self._train:
            agent.reset(seed)
            
        all_metrics = []
        steps_so_far = 0
        episode_idx = 0

        # We will manage the steps in a custom loop, so we do not call run_episode() directly.
        # Instead, we do what run_episode does, but with the additional constraint of total_steps.
        
        pbar = tqdm(total=total_steps, desc="Running steps")

        while steps_so_far < total_steps:
            episode_idx += 1
            
            # Initialize an episode
            # ep_seed = episode_idx + seed # If you want each episode to have specific seeds 
            #                           (each episode is reproducible but maybe too specific)    
            observation, info = env.reset() # seed=ep_seed
            ep_return = 0.0
            steps_in_episode = 0
            transitions = []
            terminated = False
            truncated = False
            
            # Step loop
            while not (terminated or truncated):
                # Agent selects action
                action = agent.act(observation)
                
                # Environment steps
                next_observation, reward, terminated, truncated, info = env.step(action)

                # Update reward/step counters
                ep_return += reward
                steps_in_episode += 1
                steps_so_far += 1

                # If we've hit the global step limit mid-episode, force truncation
                if steps_so_far >= total_steps:
                    truncated = True

                # Optionally store transitions
                if self._dump_transitions:
                    transitions.append((observation, action, reward, terminated, truncated))
                
                # Train the agent if needed
                if self._train:
                    agent.update(next_observation, reward, terminated, truncated)
                
                
                pbar.update(1)

                # Move to next observation
                observation = next_observation
           
            # Collect frames from the environment if needed
            frames = env.render()
            
            # Episode metrics
            metrics = {
                "ep_return": ep_return,
                "ep_length": steps_in_episode,
                "frames": frames,
                # "env_seed": ep_seed,
                "transitions": transitions,
                "agent_seed": seed,
                "episode_index": episode_idx
            }
            all_metrics.append(metrics)
            
            # Show some info in the progress bar
            pbar.set_postfix({
                "Episode": episode_idx,
                "Return": metrics["ep_return"], 
                "TotalSteps": steps_so_far
            })
            
            # Checkpointing if desired
            if self._checkpoint_freq is not None and self._checkpoint_freq != 0 and episode_idx % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Run{run_idx}_E{episode_idx}")
                agent.save(path)
        pbar.close()
        return all_metrics

    def _one_run(self, run_idx, case_num, num_episodes, total_steps, seed_offset, tuning_hp=None):
        """
        Helper for a single run: computes a seed and calls the
        appropriate single-run method.
        """
        if seed_offset is None:
            seed = random.randint(0, 2**32 - 1)
        else:
            if case_num == 1:
                seed = (run_idx - 1) * num_episodes + seed_offset
            else:
                seed = (run_idx - 1) * total_steps + seed_offset

        # build fresh env & agent each run
        env   = self._make_env()
        agent = self._make_agent(env)
        
        self.env, self.agent = env, agent
        if tuning_hp is not None:
            agent.set_hp(tuning_hp)

        if case_num == 1:
            result = self._single_run_episodes(env, agent, num_episodes, seed, run_idx)
        else:
            result = self._single_run_steps(env, agent, total_steps, seed, run_idx)
        
        # Save agent
        if self._checkpoint_freq is not None:
            path = os.path.join(self.exp_dir, f"Run{run_idx}_Last")
            agent.save(path)
        
        return result
        
    def multi_run(self, num_runs, 
                  num_episodes=0, total_steps=0,
                  seed_offset=None, 
                  dump_metrics=True, checkpoint_freq=None, 
                  dump_transitions=False, num_workers=1, tuning_hp=None):
        """
        Run multiple independent runs.
        
        Returns:
            list: A list of runs, each containing a list of episode metrics.
        """
        if (num_episodes > 0) == (total_steps > 0):
            raise ValueError("Exactly one of num_episodes or total_steps must be non-zero")
        case_num = 1 if num_episodes > 0 else 2
        
        if not (self._env_is_factory and self._agent_is_factory):
            if num_workers > 1:
                print("⚠️  Fixed env/agent instances: falling back to num_workers=1")
            num_workers = 1
            

        self._checkpoint_freq = checkpoint_freq
        self._dump_transitions = dump_transitions
        
        all_runs_metrics = []
                
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        self._one_run,
                        run_idx=run,
                        case_num=case_num,
                        num_episodes=num_episodes,
                        total_steps=total_steps,
                        seed_offset=seed_offset,
                        tuning_hp=tuning_hp,
                    ): run
                    for run in range(1, num_runs + 1)
                }
                for fut in as_completed(futures):
                    run_idx     = futures[fut]
                    run_metrics = fut.result()
                    all_runs_metrics.append(run_metrics)
                    if dump_metrics:
                        path = os.path.join(self.exp_dir, f"metrics_run{run_idx}.pkl")
                        with open(path, "wb") as f:
                            pickle.dump(run_metrics, f)

        # Serial execution
        else:
            for run in range(1, num_runs + 1):
                print(f"Starting Run {run}")
                run_metrics = self._one_run(
                    run_idx=run,
                    case_num=case_num,
                    num_episodes=num_episodes,
                    total_steps=total_steps,
                    seed_offset=seed_offset,
                    tuning_hp=tuning_hp
                )
                all_runs_metrics.append(run_metrics)
                if dump_metrics:
                    path = os.path.join(self.exp_dir, f"metrics_run{run}.pkl")
                    with open(path, "wb") as f:
                        pickle.dump(run_metrics, f)

        # Dump aggregated metrics
        if dump_metrics:
            agg_path = os.path.join(self.exp_dir, "all_metrics.pkl")
            with open(agg_path, "wb") as f:
                pickle.dump(all_runs_metrics, f)
                    
        return all_runs_metrics
    
    @classmethod
    def load_transitions(cls, exp_dir):
        """
        Load and return all transitions stored in the metrics file from a previous experiment run.

        Args:
            exp_dir (str): The directory where the metrics file is stored.

        Returns:
            list: A list of transitions collected across episodes and runs.
        """
        metrics_file = os.path.join(exp_dir, "all_metrics.pkl")
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"No metrics file found in {exp_dir}")
        with open(metrics_file, "rb") as f:
            all_runs_metrics = pickle.load(f)
        
        # Extract transitions from every episode across all runs.
        all_transitions = []
        for run in all_runs_metrics:
            episode_transitions = []
            for episode in run:
                episode_transitions.append(episode.get("transitions", []))
            all_transitions.append(episode_transitions)
        return all_transitions

    @classmethod
    def load_args(cls, exp_dir):
        """
        Load and return the environment configuration from a previous experiment run.

        Args:
            exp_dir (str): The directory where the environment file is stored.

        Returns:
            dict: The environment configuration.
        """
        args_file = os.path.join(exp_dir, "args.yaml")
        if not os.path.exists(args_file):
            raise FileNotFoundError(f"No args file found in {exp_dir}")
        with open(args_file, "r") as f:
            args_dict = yaml.safe_load(f)
        args = argparse.Namespace(**args_dict)
        return args

    @classmethod
    def load_config(cls, exp_dir):
        """
        Load and return the configuration module from a previous experiment run.

        Args:
            exp_dir (str): The directory where the environment file is stored.

        Returns:
            dict: The configuration module.
        """
        config_path = os.path.join(exp_dir, "config.py")
        spec = importlib.util.spec_from_file_location("loaded_module", config_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {config_path}")
        
        # Create a module from the spec.
        config = importlib.util.module_from_spec(spec)
        
        # Execute the module in its own namespace.
        spec.loader.exec_module(config)
        
        return config