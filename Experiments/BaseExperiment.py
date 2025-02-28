from tqdm import tqdm
import os
import pickle
import random
import shutil
import importlib.util
import yaml
import argparse

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
        
        self.env = env
        self.agent = agent
        if exp_dir is not None:
            # Copy config file
            os.makedirs(exp_dir, exist_ok=True)
            self.exp_dir = exp_dir
            if config is not None:
                shutil.copy(config, exp_dir)
        
            # Save args
            file = os.path.join(self.exp_dir, "args.yaml")
            with open(file, "w") as f:
                yaml.dump(vars(args), f)

        self._dump_transitions = False
        self._checkpoint_freq = None
        self._train = train

    def run_episode(self, seed):
        """
        Run a single episode.
        
        Args:
            seed: Seed for reproducibility.
        
        Returns:
            dict: Metrics including total_reward, steps, frames, seeds, and transitions.
        """
        observation, info = self.env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        transitions = []
        while not (terminated or truncated):
            action = self.agent.act(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            
            if self._dump_transitions:
                transitions.append((observation, action, reward, terminated, truncated))
            if self._train:
                self.agent.update(next_observation, reward, terminated, truncated)
            
            total_reward += reward
            steps += 1

            # Move to next observation
            observation = next_observation
        
        frames = self.env.render()
        return {
            "total_reward": total_reward,
            "steps": steps,
            "frames": frames,
            "env_seed": seed,
            "transitions": transitions
        }

    def _single_run_episodes(self, num_episodes, seed, n_run):
        """
        Run a specified number of episodes.
        
        Returns:
            list: Episode metrics for the run.
        """
        if self._train:
            self.agent.reset(seed)
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        for episode in pbar:
            metrics = self.run_episode(episode + seed)
            metrics["agent_seed"] = seed
            metrics["episode_index"] = episode
            all_metrics.append(metrics)
            pbar.set_postfix({
                "Reward": metrics['total_reward'], 
                "Steps": metrics['steps'],
            })
            if self._checkpoint_freq is not None and episode % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Policy_Run{n_run}_E{episode}.t")
                self.agent.save(path)
        return all_metrics
    
    def _single_run_steps(self, total_steps, seed, n_run):
        """
        Run until we have executed 'total_steps' agent-environment interactions.
        If we reach total_steps in the middle of an episode, that episode is truncated 
        immediately.

        Args:
            total_steps (int): The total number of steps to run across one or more episodes.
            seed (int): Seed for reproducibility.
            n_run (int): Index of the current run (used for checkpoint naming).
            
        Returns:
            list of dict: 
                A list of episode metrics. Each element is the dictionary returned 
                by self.run_episode(...), plus extra info (agent_seed, etc.).
        """
        if self._train:
            self.agent.reset(seed)
            
        all_metrics = []
        steps_so_far = 0
        episode_idx = 0

        # We will manage the steps in a custom loop, so we do not call run_episode() directly.
        # Instead, we do what run_episode does, but with the additional constraint of total_steps.
        
        pbar = tqdm(total=total_steps, desc="Running steps")

        while steps_so_far < total_steps:
            episode_idx += 1
            
            # Initialize an episode
            observation, info = self.env.reset(seed=(seed + episode_idx))
            episode_reward = 0.0
            steps_in_episode = 0
            transitions = []
            terminated = False
            truncated = False
            
            # Step loop
            while not (terminated or truncated):
                # Agent selects action
                action = self.agent.act(observation)
                
                # Environment steps
                next_observation, reward, terminated, truncated, info = self.env.step(action)

                # Update reward/step counters
                episode_reward += reward
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
                    self.agent.update(next_observation, reward, terminated, truncated)
                
                
                pbar.update(1)

                # Move to next observation
                observation = next_observation
                
                
            
            # Collect frames from the environment if needed
            frames = self.env.render()
            
            # Episode metrics
            metrics = {
                "total_reward": episode_reward,
                "steps": steps_in_episode,
                "frames": frames,
                "env_seed": (seed + episode_idx),
                "transitions": transitions,
                "agent_seed": seed,
                "episode_index": episode_idx
            }
            all_metrics.append(metrics)
            
            # Show some info in the progress bar
            pbar.set_postfix({
                "Episode": episode_idx,
                "Reward": metrics["total_reward"], 
                "TotalSteps": steps_so_far
            })
            
            # Checkpointing if desired
            if self._checkpoint_freq is not None and episode_idx % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Policy_Run{n_run}_E{episode_idx}.t")
                self.agent.save(path)
        pbar.close()
        return all_metrics

    def multi_run(self, num_runs, 
                  num_episodes=0, total_steps=0,
                  seed_offset=None, 
                  dump_metrics=True, checkpoint_freq=None, 
                  dump_transitions=False):
        """
        Run multiple independent runs.
        
        Returns:
            list: A list of runs, each containing a list of episode metrics.
        """
        if num_episodes > 0 and total_steps == 0:
            # we run for fix num of episodes
            case_num = 1
        elif num_episodes ==0 and total_steps > 0:
            # we run for fix num of steps
            case_num = 2
        else:
            raise ValueError("Both num episode and total steps are either 0 or not 0")


        self._checkpoint_freq = checkpoint_freq
        self._dump_transitions = dump_transitions

        all_runs_metrics = []

        for run in range(1, num_runs + 1):
            print(f"Starting Run {run}")
            if case_num == 1:
                seed = random.randint(0, 2**32 - 1) if seed_offset is None else (run - 1) * num_episodes + seed_offset
                run_metrics = self._single_run_episodes(num_episodes, seed, run)
            elif case_num == 2:
                seed = random.randint(0, 2**32 - 1) if seed_offset is None else (run - 1) * total_steps + seed_offset
                run_metrics = self._single_run_steps(total_steps, seed, run)

            all_runs_metrics.append(run_metrics)
            if dump_metrics:
                file = os.path.join(self.exp_dir, "metrics.pkl")
                with open(file, "wb") as f:
                    pickle.dump(all_runs_metrics, f)
                path = os.path.join(self.exp_dir, f"Policy_Run{run}_Last.t")
                self.agent.save(path)
                    
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
        metrics_file = os.path.join(exp_dir, "metrics.pkl")
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