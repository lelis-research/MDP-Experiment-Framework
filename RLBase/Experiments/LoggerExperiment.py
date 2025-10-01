from torch.utils.tensorboard import SummaryWriter 
import os
from tqdm import tqdm
import pickle
import random
import numpy as np

from . import BaseExperiment

class call_back:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_counter = 0

    def __call__(self, data_dict, tag, counter=None):
        if counter is None:
            counter = self.global_counter
            self.global_counter += 1

        for key in data_dict:
            self.writer.add_scalar(f"{tag}/{key}", data_dict[key], counter)
    
    def reset(self):
        self.global_counter = 0
    
    def close(self):
        self.writer.close()
    
        
    

class LoggerExperiment(BaseExperiment):
    """
    An experiment class that runs episodes, collects metrics, and logs them to TensorBoard.
    """
    def __init__(self, env, agent, exp_dir, train=True, config=None, args=None):
        super().__init__(env, agent, exp_dir, train=train, config=config, args=args)
        self.call_back = call_back(log_dir=exp_dir)
    
    def _single_run_episodes(self, env, agent, num_episodes, seed, run_idx):
        """
        Run a series of episodes for one experimental run and log metrics to TensorBoard.
        
        Args:
            num_episodes (int): Number of episodes in the run.
            seed (int): Base seed for reproducibility.
            run_idx (int): Run identifier.
        
        Returns:
            list: A list of episode metrics.
        """
        best_agent, best_return = None, -np.inf
        self.call_back.reset()
        if self._train:
            agent.reset(seed)
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        for episode_idx in pbar:
            frames = []
            infos = []
            agent_logs = []
            # Use a seed to ensure reproducibility.
            # ep_seed = episode_idx + seed
            observation, info = env.reset() # seed=ep_seed
            if env.render_mode == "human":
                env.render()
            elif env.render_mode == "ansi":
                frames.append(env.render())
                    
            ep_return = 0.0
            steps = 0
            terminated = False
            truncated = False
            transitions = []

            while not (terminated or truncated):
                action = agent.act(observation, greedy=not self._train)
                next_observation, reward, terminated, truncated, info = env.step(action)
                infos.append(info)

                if self._dump_transitions:
                    transitions.append((observation, action, reward, terminated, truncated))
                if self._train:
                    # Pass history to callback
                    agent.update(next_observation, reward, terminated, truncated,
                                    call_back=lambda data: self.call_back(data, f"agents/run_{run_idx}"))

                ep_return += info["actual_reward"] if "actual_reward" in info else reward
                steps += 1
                observation = next_observation
                
                if env.render_mode == "human":
                    env.render()
                elif env.render_mode == "ansi":
                    frames.append(env.render())
                    
                if hasattr(agent, 'log'):
                    agent_logs.append(agent.log())
                

            if env.render_mode == "rgb_array_list":
                frames = env.render()
            
                
            metrics = {
                "ep_return":    ep_return,
                "ep_length":    steps,
                "frames":       frames,
                "infos": infos,
                # "env_seed":     ep_seed,
                "transitions":  transitions,
                "agent_seed": seed,
                "episode_index": episode_idx,
                "agent_logs": agent_logs,
            }
            all_metrics.append(metrics)
            if ep_return >= best_return:
                best_return = ep_return
                best_agent = agent.save()

            self.call_back({"ep_return": metrics["ep_return"]},
                           f"ep_return/run_{run_idx}", 
                           episode_idx)
            self.call_back({"ep_length": metrics["ep_length"]},
                           f"ep_length/run_{run_idx}", 
                           episode_idx)
            
            pbar.set_postfix({
                "Return": metrics['ep_return'], 
                "Steps": metrics['ep_length'],
            })
            if self._checkpoint_freq is not None and self._checkpoint_freq != 0 and episode_idx % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Run{run_idx}_E{episode_idx}")
                agent.save(path)
        return all_metrics, best_agent
    
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
        best_agent, best_return = None, -np.inf
        self.call_back.reset()
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
            frames = []
            agent_logs = []

            
            # Initialize an episode
            # ep_seed = episode_idx + seed
            observation, info = env.reset() #seed=ep_seed)
            if env.render_mode == "human":
                env.render()
            elif env.render_mode == "ansi":
                frames.append(env.render())
                
            ep_return = 0.0
            steps_in_episode = 0
            transitions = []
            infos = []
            terminated = False
            truncated = False
            
            # Step loop
            while not (terminated or truncated):
                # Agent selects action
                action = agent.act(observation, greedy=not self._train)
                
                # Environment steps
                next_observation, reward, terminated, truncated, info = env.step(action)
                infos.append(info)
                
                # Update reward/step counters
                ep_return += info["actual_reward"] if "actual_reward" in info else reward
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
                    agent.update(next_observation, reward, terminated, truncated, 
                                      call_back=lambda data_dict: self.call_back(data_dict, f"agents/run_{run_idx}"))
                
                pbar.update(1)
                # Move to next observation
                observation = next_observation
                
                if env.render_mode == "human":
                    env.render()
                elif env.render_mode == "ansi":
                    frames.append(env.render())
                
                if hasattr(agent, 'log'):
                    agent_logs.append(agent.log())
                
            # Collect frames from the environment if needed
            if env.render_mode == "rgb_array_list":
                frames = env.render()
            
            # Episode metrics
            metrics = {
                "ep_return": ep_return,
                "ep_length": steps_in_episode,
                "frames": frames,
                "infos": infos,
                # "env_seed": ep_seed,
                "transitions": transitions,
                "agent_seed": seed,
                "episode_index": episode_idx,
                "agent_logs": agent_logs,
            }
            all_metrics.append(metrics)
            
            if ep_return >= best_return:
                best_return = ep_return
                best_agent = agent.save()

            self.call_back({"ep_return": metrics["ep_return"]},
                           f"ep_return/run_{run_idx}", 
                           steps_so_far)
            self.call_back({"ep_length": metrics["ep_length"]},
                           f"ep_length/run_{run_idx}", 
                           steps_so_far)
            
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
        return all_metrics, best_agent

    def multi_run(self, num_runs, 
                  num_episodes=0, total_steps=0, 
                  seed_offset=None, 
                  dump_metrics=True, checkpoint_freq=None, 
                  dump_transitions=False, num_workers=1, tuning_hp=None):
        """
        Run multiple experimental runs, log metrics, and save checkpoints.
        
        Args:
            num_runs (int): Number of independent runs.
            num_episodes (int): Number of episodes per run.
            seed_offset (int, optional): Offset seed for reproducibility.
            dump_metrics (bool): Whether to dump metrics to disk.
            checkpoint_freq (int, optional): Frequency to save checkpoints.
            dump_transitions (bool): Whether to dump raw transitions.
        
        Returns:
            list: A list containing metrics for all runs.
        """
        all_runs_metrics = super().multi_run(num_runs, 
                                             num_episodes, total_steps,
                                             seed_offset, 
                                             dump_metrics, checkpoint_freq, 
                                             dump_transitions, num_workers,
                                             tuning_hp=tuning_hp)
        self.call_back.close()
        return all_runs_metrics