from torch.utils.tensorboard import SummaryWriter 
import os
from tqdm import tqdm
import pickle
import random

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
    
    def run_episode(self, seed, call_back=None):
        """
        Run a single episode.
        
        Args:
            seed: Seed for reproducibility.
            call_back (function, optional): Callback to log intermediate metrics.
        
        Returns:
            dict: Episode metrics including total_reward, steps, frames, env_seed, and transitions.
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
                self.agent.update(next_observation, reward, terminated, truncated, call_back)
            
            total_reward += reward
            steps += 1

            # Move to next observation
            observation = next_observation
        
        frames = self.env.render()
        return {"total_reward": total_reward, "steps": steps, 
                "frames": frames, "env_seed": seed, "transitions": transitions}
    
    def _single_run_episodes(self, num_episodes, seed, n_run):
        """
        Run a series of episodes for one experimental run and log metrics to TensorBoard.
        
        Args:
            num_episodes (int): Number of episodes in the run.
            seed (int): Base seed for reproducibility.
            n_run (int): Run identifier.
        
        Returns:
            list: A list of episode metrics.
        """
        self.call_back.reset()
        if self._train:
            self.agent.reset(seed)
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        for episode in pbar:
            # Use a seed to ensure reproducibility.
            metrics = self.run_episode(episode + seed,
                                       call_back=lambda data_dict: self.call_back(data_dict, f"agents/run_{n_run}"))
            metrics['agent_seed'] = seed
            metrics['episode_index'] = episode
            all_metrics.append(metrics)

            self.call_back({"total_reward": metrics["total_reward"]},
                           f"total_reward/run_{n_run}", 
                           episode)
            self.call_back({"num_steps": metrics["steps"]},
                           f"num_steps/run_{n_run}", 
                           episode)
            
            pbar.set_postfix({
                "Reward": metrics['total_reward'], 
                "Steps": metrics['steps'],
            })
            if self._checkpoint_freq is not None and episode % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Policy_Run{n_run}_E{episode}")
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
        self.call_back.reset()
        if self._train:
            self.agent.reset(seed)
            
        all_metrics = []
        steps_so_far = 0
        episode = 0

        # We will manage the steps in a custom loop, so we do not call run_episode() directly.
        # Instead, we do what run_episode does, but with the additional constraint of total_steps.
        
        pbar = tqdm(total=total_steps, desc="Running steps")

        while steps_so_far < total_steps:
            episode += 1
            
            # Initialize an episode
            observation, info = self.env.reset(seed=(seed + episode))
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
                    self.agent.update(next_observation, reward, terminated, truncated, 
                                      call_back=lambda data_dict: self.call_back(data_dict, f"agents/run_{n_run}"))
                
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
                "env_seed": (seed + episode),
                "transitions": transitions,
                "agent_seed": seed,
                "episode_index": episode
            }
            all_metrics.append(metrics)

            self.call_back({"total_reward": metrics["total_reward"]},
                           f"total_reward/run_{n_run}", 
                           episode)
            self.call_back({"num_steps": metrics["steps"]},
                           f"num_steps/run_{n_run}", 
                           episode)
            
            # Show some info in the progress bar
            pbar.set_postfix({
                "Episode": episode,
                "Reward": metrics["total_reward"], 
                "TotalSteps": steps_so_far
            })
            
            # Checkpointing if desired
            if self._checkpoint_freq is not None and episode % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Policy_Run{n_run}_E{episode}.t")
                self.agent.save(path)
        pbar.close()
        return all_metrics

    def multi_run(self, num_runs, 
                  num_episodes=0, total_steps=0, 
                  seed_offset=None, 
                  dump_metrics=True, checkpoint_freq=None, 
                  dump_transitions=False):
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
                                             dump_transitions)
        self.call_back.close()
        return all_runs_metrics