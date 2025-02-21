from tqdm import tqdm
import os
import pickle
import random

class BaseExperiment:
    """
    Base class for running episodes and collecting metrics.
    """
    def __init__(self, env, agent, exp_dir=None, train=True):
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
            os.makedirs(exp_dir, exist_ok=True)
            self.exp_dir = exp_dir

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
            observation, reward, terminated, truncated, info = self.env.step(action)
            if self._dump_transitions:
                transitions.append((observation, reward, terminated, truncated))
            if self._train:
                self.agent.update(observation, reward, terminated, truncated)
            total_reward += reward
            steps += 1
        
        frames = self.env.render()
        return {
            "total_reward": total_reward,
            "steps": steps,
            "frames": frames,
            "env_seed": seed,
            "transitions": transitions
        }

    def _single_run(self, num_episodes, seed, n_run):
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
            all_metrics.append(metrics)
            pbar.set_postfix({
                "Reward": metrics['total_reward'], 
                "Steps": metrics['steps'],
            })
            if self._checkpoint_freq is not None and episode % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Policy_Run{n_run}_E{episode}.t")
                self.agent.save(path)
        return all_metrics

    def multi_run(self, num_runs, num_episodes, seed_offset=None, 
                  dump_metrics=True, checkpoint_freq=None, 
                  dump_transitions=False):
        """
        Run multiple independent runs.
        
        Returns:
            list: A list of runs, each containing a list of episode metrics.
        """
        self._checkpoint_freq = checkpoint_freq
        self._dump_transitions = dump_transitions

        all_runs_metrics = []

        # Save environment configuration for repeatability.
        file = os.path.join(self.exp_dir, "env.pkl")
        with open(file, "wb") as f:
            pickle.dump(self.env.custom_config, f)

        for run in range(1, num_runs + 1):
            print(f"Starting Run {run}")
            seed = random.randint(0, 2**32 - 1) if seed_offset is None else (run - 1) * num_episodes + seed_offset
            run_metrics = self._single_run(num_episodes, seed, run)
            all_runs_metrics.append(run_metrics)
            if dump_metrics:
                file = os.path.join(self.exp_dir, "metrics.pkl")
                with open(file, "wb") as f:
                    pickle.dump(all_runs_metrics, f)
                path = os.path.join(self.exp_dir, f"Policy_Run{run}_Last.t")
                self.agent.save(path)
                    
        return all_runs_metrics