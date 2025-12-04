from tqdm import tqdm
import os
import pickle
import random
import shutil
import importlib.util
import yaml
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from .CallBacks import JsonlCallback, TBCallBack, EmptyCallBack
        
class OnlineTrainer:
    """
    Base class for running episodes and collecting metrics.
    For Vectorized Env
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
        self._dump_actions = True
        self._train = train
        
        
        
        self.call_back = TBCallBack(log_dir=exp_dir, flush_every=100) #makes it super slow on compute canada
        
        # self.call_back = EmptyCallBack(log_dir=exp_dir)
        # self.call_back = JsonlCallback(log_path=os.path.join(exp_dir, "metrics.jsonl"), flush_every=100)
        
    @staticmethod
    def get_single_observation(observation, index):
        if isinstance(observation, dict):
            return {k: (v[index] if hasattr(v, "__getitem__") else v) for k, v in observation.items()}
        elif hasattr(observation, "__getitem__"):
            return observation[index]
        else:
            raise NotImplementedError(f"Vectorized Observation type {type(observation)} is not defined")
    
    def extract_actual_rewards(self, info, reward):
        if isinstance(info, (list, tuple)) and len(info) == len(reward):
            return np.asarray(
                [info.get("actual_reward", reward[j]) if isinstance(info, dict) else reward[j]
                for j, info in enumerate(info)],
                dtype=np.float32,
            )
        if isinstance(info, dict) and "actual_reward" in info:
            ar = np.asarray(info["actual_reward"], dtype=np.float32)
            if ar.shape == reward.shape:
                return ar
        return reward

    def _single_run_episodes(self, env, agent, num_episodes, seed, run_idx):
        """
        NOTE: **** env is vectorized *****
        Run a specified number of episodes.
        
        Returns:
            list: Episode metrics for the run.
        """
        best_agent, best_return = None, -np.inf
        self.call_back.reset()
        if self._train:
            agent.reset(seed)
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        num_envs = env.num_envs
        agent.set_num_env(num_envs)
        
        ep_returns = np.zeros(num_envs, dtype=np.float32)
        ep_lengths = np.zeros(num_envs, dtype=np.int64)
        
        transitions = [[] for _ in range(num_envs)]
        actions_log = [[] for _ in range(num_envs)]
        agent_logs = [[] for _ in range(num_envs)]
        
        seeds = [seed + i for i in range(num_envs)]
        observation, info = env.reset(seed=seeds)
        frames = [[] for _ in range(num_envs)]
        
        if env.render_mode == "human":
            env.envs[0].render()
        elif env.render_mode == "ansi" or env.render_mode == "rgb_array":
            for i, fr in enumerate(env.render()):
                frames[i].append(fr)
        
        episodes_done = 0
        steps_so_far = 0
        
        checkpoint_counter = 0
        
        while episodes_done < num_episodes:
            action = agent.act(observation, greedy=not self._train)
            
            if hasattr(agent, 'log'):
                log_entry = agent.log()
                for i in range(num_envs):
                    agent_logs[i].append(log_entry)
            
            next_observation, reward, terminated, truncated, info = env.step(action)  
            
            if env.render_mode == "human":
                env.envs[0].render()
            elif env.render_mode == "ansi" or env.render_mode == "rgb_array":
                for i, fr in enumerate(env.render()):
                    frames[i].append(fr)
            
            
            if self._dump_transitions:
                for i in range(num_envs):
                    transitions[i].append((self.get_single_observation(observation, i), 
                                           action[i], reward[i],
                                           bool(terminated[i]), bool(truncated[i])))
            if self._dump_actions:
                for i in range(num_envs):
                    actions_log[i].append(action[i])
            
            ep_returns += self.extract_actual_rewards(info, reward)
            ep_lengths += 1
            steps_so_far += num_envs
            
            if self._train:
                agent.update(next_observation, reward, terminated, truncated,
                            call_back=lambda data_dict: self.call_back(data_dict, f"agents/run_{run_idx}", steps_so_far))
                
            
            observation = next_observation
            
            dones = np.logical_or(terminated, truncated)

            for i in range(len(dones)):
                if not dones[i]:
                    continue
                episodes_done += 1

                if env.render_mode == "rgb_array_list":
                    frames[i] = env.envs[i].render()
                    
                
                metrics = {
                    "ep_return": float(ep_returns[i]),
                    "ep_length": int(ep_lengths[i]),
                    "frames": frames[i],
                    "transitions": transitions[i],
                    "actions": actions_log[i],
                    "agent_seed": seed,
                    "episode_index": episodes_done,
                    "agent_logs": agent_logs[i]
                }
                all_metrics.append(metrics)

                # Best agent snapshot (checkpoint dict from agent.save())
                if ep_returns[i] >= best_return:
                    best_return = ep_returns[i]
                    best_agent = agent.save()

                self.call_back({"ep_return": metrics["ep_return"]},
                           f"ep_return/run_{run_idx}", 
                           episodes_done,
                           force=True)
                self.call_back({"ep_length": metrics["ep_length"]},
                            f"ep_length/run_{run_idx}", 
                            episodes_done,
                            force=True)
            
                # Progress + optional checkpoint
                pbar.update(1)
                pbar.set_postfix({"Return": metrics["ep_return"], 
                                  "Steps": metrics["ep_length"]})

                checkpoint_counter += 1
                if self._checkpoint_freq is not None and self._checkpoint_freq != 0 and checkpoint_counter >= self._checkpoint_freq:
                    path = os.path.join(self.exp_dir, f"Run{run_idx}_E{episodes_done}")
                    agent.save(path)
                    checkpoint_counter = 0

                # Reset per-sub-env trackers for that env slot
                ep_returns[i] = 0.0
                ep_lengths[i] = 0
                if self._dump_transitions:
                    transitions[i] = []
                if self._dump_actions:
                    actions_log[i] = []
                
                agent_logs[i] = []
                frames[i] = []
           
            
        pbar.close()
        return all_metrics, best_agent
    
    def _single_run_steps(self, env, agent, total_steps, seed, run_idx):
        """
        NOTE: **** env is vectorized *****
        Run a specified number of env steps in total.
        
        Returns:
            list: Episode metrics for the run.
        """
        best_agent, best_return = None, -np.inf
        self.call_back.reset()
        if self._train:
            agent.reset(seed)
            
        all_metrics = []
        pbar = tqdm(total=total_steps, desc="Running steps")
        num_envs = env.num_envs
        agent.set_num_env(num_envs)

        ep_returns = np.zeros(num_envs, dtype=np.float32)
        ep_lengths = np.zeros(num_envs, dtype=np.int64)
        
        transitions = [[] for _ in range(num_envs)]
        actions_log = [[] for _ in range(num_envs)]
        agent_logs = [[] for _ in range(num_envs)]
        
        seeds = [seed + i for i in range(num_envs)]
        observation, info = env.reset(seed=seeds)
        frames = [[] for _ in range(num_envs)]
        
        
        if env.render_mode == "human":
            env.envs[0].render()
        elif env.render_mode == "ansi":
            for i, fr in enumerate(env.render()):
                frames[i].append(fr)
                
        steps_so_far = 0
        episodes_done = 0
        
        checkpoint_counter = 0
        
        while steps_so_far < total_steps:
            action = agent.act(observation, greedy=not self._train)
            
            if hasattr(agent, 'log'):
                log_entry = agent.log()
                for i in range(num_envs):
                    agent_logs[i].append(log_entry)
            
            next_observation, reward, terminated, truncated, info = env.step(action)  
    
            if env.render_mode == "human":
                env.envs[0].render()
            elif env.render_mode == "ansi":
                for i, fr in enumerate(env.render()):
                    frames[i].append(fr)
            
            if self._dump_transitions:
                for i in range(num_envs):
                    transitions[i].append((self.get_single_observation(observation, i), 
                                        action[i], reward[i],
                                        bool(terminated[i]), bool(truncated[i])))
            if self._dump_actions:
                for i in range(num_envs):
                    actions_log[i].append(action[i])
                    
            ep_returns += self.extract_actual_rewards(info, reward)
            
            ep_lengths += 1
            steps_so_far += num_envs  # aggregate interactions across sub-envs
            
            # if steps_so_far >= total_steps:
            #     truncated = np.ones_like(truncated)
                    
            if self._train:
                agent.update(next_observation, reward, terminated, truncated, 
                            call_back=lambda data_dict: self.call_back(data_dict, f"agents/run_{run_idx}", steps_so_far))
            
            pbar.update(num_envs)
            observation = next_observation
            
            checkpoint_counter += num_envs
            if self._checkpoint_freq is not None and self._checkpoint_freq != 0 and checkpoint_counter >= self._checkpoint_freq:
                path = os.path.join(self.exp_dir, f"Run{run_idx}_E{steps_so_far}")
                agent.save(path)
                checkpoint_counter = 0
            
            dones = np.logical_or(terminated, truncated)
            for i in range(len(dones)):
                if not dones[i]:
                    continue
                episodes_done += 1
                
                if env.render_mode == "rgb_array_list":
                    frames[i] = env.envs[i].render()
                

                metrics = {
                    "ep_return": float(ep_returns[i]),
                    "ep_length": int(ep_lengths[i]),
                    "frames": frames[i],
                    "transitions": transitions[i],
                    "actions": actions_log[i],
                    "agent_seed": seed,
                    "episode_index": episodes_done,
                    "agent_logs": agent_logs[i]
                }
                all_metrics.append(metrics)
                
                # Best agent snapshot (checkpoint dict from agent.save())
                if ep_returns[i] >= best_return:
                    best_return = ep_returns[i]
                    best_agent = agent.save()
                    
                self.call_back({"ep_return": metrics["ep_return"]},
                           f"ep_return/run_{run_idx}", 
                           steps_so_far,
                           force=True)
                self.call_back({"ep_length": metrics["ep_length"]},
                            f"ep_length/run_{run_idx}", 
                            steps_so_far,
                            force=True)

                
                pbar.set_postfix({"Episode": episodes_done, 
                                  "Return": metrics["ep_return"], 
                                  "TotalSteps": steps_so_far})

                # Reset per-sub-env trackers for that env slot
                ep_returns[i] = 0.0
                ep_lengths[i] = 0
                if self._dump_transitions:
                    transitions[i] = []
                if self._dump_actions:
                    actions_log[i] = []
                
                agent_logs[i] = []
                frames[i] = []


        pbar.close()
        return all_metrics, best_agent

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
        
        print("\n" + "=" * 70)
        print(f"🎯 Starting Run {run_idx} | Seed: {seed}")
        print("-" * 70)
        print(f"🌍 Environment        : {env}")
        print(f"  └─ Spec            : {getattr(env.envs[0], 'spec', env.envs[0].__class__.__name__)}")
        print(f"📥 Observation Space : {env.single_observation_space}")
        print(f"🎮 Action Space      : {env.single_action_space}")
        print(f"🤖 Agent             : {agent}")
        print(f"💻 Device            : {agent.device}")
        # print("=" * 70 + "\n")
        
        self.env, self.agent = env, agent
        if tuning_hp is not None:
            agent.set_hp(tuning_hp)

        if case_num == 1:
            result, best_agent = self._single_run_episodes(env, agent, num_episodes, seed, run_idx)
        else:
            result, best_agent = self._single_run_steps(env, agent, total_steps, seed, run_idx)
        
        env.close()
        
        # Save last and best agent
        if self._checkpoint_freq is not None:
            path = os.path.join(self.exp_dir, f"Run{run_idx}_Last")
            agent.save(path)
            
            path = os.path.join(self.exp_dir, f"Run{run_idx}_Best")
            torch.save(best_agent, f"{path}_agent.t")
        
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
        
        self.call_back.close()      
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
            raise FileNotFoundError(f"No args.yaml file found in {exp_dir}")
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

