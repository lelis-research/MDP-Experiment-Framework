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
    This class handles running episodes and collecting metrics.
    It will save the results on TensorBoard
    """
    def __init__(self, env, agent, exp_dir, train=True):
        super().__init__(env, agent, exp_dir, train=train)
        self.call_back = call_back(log_dir=exp_dir)
    
    def run_episode(self, seed, call_back=None):
        """
        Run a single episode.
        
        Args:
            seed: (Optional) Random seed for the episode.
        
        Returns:
            A dictionary containing metrics such as total_reward and steps.
        """
        observation, info = self.env.reset(seed=seed) 
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        transitions = []
        while not (terminated or truncated):
            action = self.agent.act(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            if self._dump_transitions:
                transitions.append((observation, reward, terminated, truncated))
            if self._train:
                self.agent.update(observation, reward, terminated, truncated, call_back)
            
            total_reward += reward
            steps += 1
        
        frames = self.env.render()
        return {"total_reward": total_reward, "steps": steps, 
                "frames":frames, "env_seed": seed, "transitions": transitions}
    
    def _single_run(self, num_episodes, seed, n_run):
        """
        Run the experiment for a specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to run.
            seed_offset (int): An offset for the seed.
            writer (SummaryWriter, optional): A TensorBoard writer to log this run's metrics.
                                               If not provided, uses self.writer.
        
        Returns:
            A list of episode metrics for analysis.
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
            all_metrics.append(metrics)


            self.call_back({"total_reward": metrics["total_reward"]},
                           f"total_reward/run_{n_run}", 
                           episode)
            self.call_back({"num_steps": metrics["steps"]},
                           f"num_steps/run_{n_run}", 
                           episode)
            
            
            # Update the progress bar.
            pbar.set_postfix({
                "Reward": metrics['total_reward'], 
                "Steps": metrics['steps'],
            })
            if self._checkpoint_freq is not None and episode % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Policy_Run{n_run}_E{episode}")
                self.agent.save(path)
        return all_metrics

    def multi_run(self, num_runs, num_episodes, seed_offset=None, 
                  dump_metrics=True, checkpoint_freq=None, 
                  dump_transitions=False):
        
        all_runs_metrics = super().multi_run(num_runs, num_episodes, seed_offset, 
                                             dump_metrics, checkpoint_freq, 
                                             dump_transitions)
        self.call_back.close()
            
        return all_runs_metrics