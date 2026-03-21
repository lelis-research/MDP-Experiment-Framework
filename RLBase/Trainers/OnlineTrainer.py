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
from collections import deque
import time

from .CallBacks import JsonlCallback, TBCallBack, EmptyCallBack

        
class OnlineTrainer:
    """
    Base class for running episodes/steps and collecting metrics with vectorized environments.
    """
    def __init__(self, env, agent, exp_dir=None, train=True, config=None, args=None):
        """
        Args:
            env: An initialized environment or a factory that returns one.
            agent: An agent instance or a factory that accepts an env and returns one.
            exp_dir (str, optional): Directory to save checkpoints and metrics; created if missing.
            train (bool): Whether to train the agent (False -> greedy acting, no updates).
            config (str, optional): Path to a config file to copy into exp_dir.
            args (Namespace, optional): Parsed CLI args; stored as args.yaml in exp_dir.
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
            
            
        self.exp_dir = exp_dir
        if exp_dir is not None:
            # Copy config file
            os.makedirs(exp_dir, exist_ok=True)
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
        self._dump_metrics = True
        self._metrics_dump_every = 50   # flush every 50 finished episodes
        self._keep_metrics_in_memory = False
        
        # --- Timing stats ---
        self._timing_log_every = 2000  # steps (aggregated across envs) between logs

        self._t_act_total = 0.0
        self._t_upd_total = 0.0
        self._t_act_count = 0
        self._t_upd_count = 0

        # optional: recent-window moving averages (smoother)
        self._act_window = deque(maxlen=200)
        self._upd_window = deque(maxlen=200)
        
        
        self.call_back = TBCallBack(log_dir=exp_dir, flush_every=10, flush_mode="raw") #makes it super slow on compute canada
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
            agent.train()
        else:
            agent.eval()
            
        metrics_buffer = []
        all_metrics = [] if self._keep_metrics_in_memory else None
        
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
            t0 = time.perf_counter()
            action = agent.act(observation)
            dt = time.perf_counter() - t0
            
            # update the timers for log
            self._t_act_total += dt
            self._t_act_count += 1
            self._act_window.append(dt)
            
            
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
            
            t0 = time.perf_counter()
            agent.update(next_observation, reward, terminated, truncated,
                        call_back=lambda data_dict: self.call_back(data_dict, f"agents/run_{run_idx}", steps_so_far))
            dt = time.perf_counter() - t0
            self._t_upd_total += dt
            self._t_upd_count += 1
            self._upd_window.append(dt)
            
            if self._timing_log_every and (steps_so_far % self._timing_log_every) < num_envs:
                self._log_timing(run_idx, steps_so_far, num_envs)
            
            if hasattr(agent, 'log'):
                log_entries = agent.log()  # list length num_envs
                for env_i in range(num_envs):
                    entry = log_entries[env_i]
                    if entry is not None:
                        agent_logs[env_i].append(entry)
            
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
                    "agent_logs": self.concat_dicts_of_arrays(agent_logs[i], axis=0)
                }
                metrics_buffer.append(metrics)

                if self._keep_metrics_in_memory:
                    all_metrics.append(metrics)

                if self._dump_metrics and len(metrics_buffer) >= self._metrics_dump_every:
                    self._append_metrics_chunk(run_idx, metrics_buffer)
                    metrics_buffer.clear()
    

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
        self._flush_metrics_buffer(run_idx, metrics_buffer)
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
            agent.train()
        else:
            agent.eval()
            
        metrics_buffer = []
        all_metrics = [] if self._keep_metrics_in_memory else None
        
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
        elif env.render_mode == "ansi" or env.render_mode == "rgb_array":
            for i, fr in enumerate(env.render()):
                frames[i].append(fr)

        steps_so_far = 0
        episodes_done = 0
        
        checkpoint_counter = 0
        
        while steps_so_far < total_steps:
            t0 = time.perf_counter()
            action = agent.act(observation)
            dt = time.perf_counter() - t0
            
            # update the timers for log
            self._t_act_total += dt
            self._t_act_count += 1
            self._act_window.append(dt)
            
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
            steps_so_far += num_envs  # aggregate interactions across sub-envs
            
            # if steps_so_far >= total_steps:
            #     truncated = np.ones_like(truncated)
                    
            t0 = time.perf_counter()
            agent.update(next_observation, reward, terminated, truncated,
                        call_back=lambda data_dict: self.call_back(data_dict, f"agents/run_{run_idx}", steps_so_far))
            dt = time.perf_counter() - t0
            self._t_upd_total += dt
            self._t_upd_count += 1
            self._upd_window.append(dt)
            
            if self._timing_log_every and (steps_so_far % self._timing_log_every) < num_envs:
                self._log_timing(run_idx, steps_so_far, num_envs)
            
            if hasattr(agent, 'log'):
                log_entries = agent.log()  # list length num_envs
                for env_i in range(num_envs):
                    entry = log_entries[env_i]
                    if entry is not None:
                        agent_logs[env_i].append(entry)
            
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
                    "agent_logs": self.concat_dicts_of_arrays(agent_logs[i], axis=0)
                }
                metrics_buffer.append(metrics)

                if self._keep_metrics_in_memory:
                    all_metrics.append(metrics)

                if self._dump_metrics and len(metrics_buffer) >= self._metrics_dump_every:
                    self._append_metrics_chunk(run_idx, metrics_buffer)
                    metrics_buffer.clear()
                
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
        self._flush_metrics_buffer(run_idx, metrics_buffer)
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

        if self._dump_metrics:
            self._reset_metrics_stream(run_idx)
    
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
                  dump_transitions=False, num_workers=1, tuning_hp=None,
                  metrics_dump_every=50, keep_metrics_in_memory=False):
        """
        Run multiple independent runs (either by episode count or total step budget).
        
        Args:
            num_runs (int): Number of independent runs.
            num_episodes (int): Episodes per run (mutually exclusive with total_steps).
            total_steps (int): Total env steps per run (mutually exclusive with num_episodes).
            seed_offset (int or None): Optional offset for deterministic seeding.
            dump_metrics (bool): Whether to pickle per-run and aggregated metrics.
            checkpoint_freq (int or None): If set, saves agent snapshots every N steps/episodes.
            dump_transitions (bool): If True, store per-step transitions in metrics.
            num_workers (int): Threaded parallelism over runs when env/agent are factories.
            tuning_hp: Optional hyperparameters passed to agent.set_hp.
        
        Returns:
            list: A list of runs, each containing a list of episode metrics.
        """
        if (num_episodes > 0) == (total_steps > 0):
            raise ValueError("Exactly one of num_episodes or total_steps must be non-zero")
        if dump_metrics and self.exp_dir is None:
            raise ValueError("dump_metrics=True requires exp_dir to be set in the constructor")
        if checkpoint_freq is not None and self.exp_dir is None:
            raise ValueError("checkpoint_freq requires exp_dir to be set in the constructor")
        case_num = 1 if num_episodes > 0 else 2
        
        if not (self._env_is_factory and self._agent_is_factory):
            if num_workers > 1:
                print("⚠️  Fixed env/agent instances: falling back to num_workers=1")
            num_workers = 1
            

        self._checkpoint_freq = checkpoint_freq
        self._dump_transitions = dump_transitions
        self._dump_metrics = dump_metrics
        self._metrics_dump_every = metrics_dump_every
        self._keep_metrics_in_memory = keep_metrics_in_memory

        
        all_runs_metrics = [None] * num_runs if keep_metrics_in_memory else None

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
                    
                    if keep_metrics_in_memory:
                        all_runs_metrics[run_idx - 1] = run_metrics
                    

        # Serial execution
        else:
            for run_idx in range(1, num_runs + 1):
                run_metrics = self._one_run(
                    run_idx=run_idx,
                    case_num=case_num,
                    num_episodes=num_episodes,
                    total_steps=total_steps,
                    seed_offset=seed_offset,
                    tuning_hp=tuning_hp
                )
                
                if keep_metrics_in_memory:
                    all_runs_metrics[run_idx - 1] = run_metrics
                

        # Dump aggregated metrics
        if dump_metrics and keep_metrics_in_memory:
            agg_path = os.path.join(self.exp_dir, "all_metrics.pkl")
            with open(agg_path, "wb") as f:
                pickle.dump(all_runs_metrics, f)
        
        self.call_back.close()      
        return all_runs_metrics
    
    def _log_timing(self, run_idx: int, steps_so_far: int, num_envs: int):
        if self._t_act_count == 0 or self._t_upd_count == 0:
            return

        act_avg_iter = self._t_act_total / self._t_act_count
        upd_avg_iter = self._t_upd_total / self._t_upd_count

        # per env-step
        act_avg_step = act_avg_iter / max(num_envs, 1)
        upd_avg_step = upd_avg_iter / max(num_envs, 1)

        # moving window
        act_ma = (sum(self._act_window) / len(self._act_window)) if self._act_window else act_avg_iter
        upd_ma = (sum(self._upd_window) / len(self._upd_window)) if self._upd_window else upd_avg_iter

        data = {
            "time_act_avg_iter_s": act_avg_iter,
            "time_update_avg_iter_s": upd_avg_iter,
            "time_act_avg_envstep_s": act_avg_step,
            "time_update_avg_envstep_s": upd_avg_step,
            "time_act_ma_iter_s": act_ma,
            "time_update_ma_iter_s": upd_ma,
        }

        # log to your callback (TB/Jsonl/etc.)
        self.call_back(data, f"timing/run_{run_idx}", steps_so_far, force=True)

        # optional: print occasionally
        # print(
        #     f"[timing run {run_idx} @ steps {steps_so_far}] "
        #     f"act {act_avg_iter*1e3:.3f}ms/iter ({act_avg_step*1e3:.3f}ms/envstep), "
        #     f"upd {upd_avg_iter*1e3:.3f}ms/iter ({upd_avg_step*1e3:.3f}ms/envstep)"
        # )
        
    @classmethod
    def load_transitions(cls, exp_dir):
        """
        Load transitions from all per-run streamed metric files.
        Returns:
            list[run][episode] = transitions
        """
        all_runs_metrics = cls.load_all_run_metrics(exp_dir)

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
        Load and return the saved CLI args Namespace from a previous experiment run.

        Args:
            exp_dir (str): The directory where the args.yaml file is stored.

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

    @classmethod
    def load_run_metrics(cls, exp_dir, run_idx):
        """
        Load a streamed metrics file for one run.
        Supports files written via repeated pickle.dump(..., 'ab').
        Returns a flat list of episode metric dicts.
        """
        path = os.path.join(exp_dir, f"metrics_run{run_idx}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No metrics file found for run {run_idx} in {exp_dir}")

        all_metrics = []
        with open(path, "rb") as f:
            while True:
                try:
                    chunk = pickle.load(f)
                except EOFError:
                    break

                if isinstance(chunk, list):
                    all_metrics.extend(chunk)
                else:
                    raise TypeError(
                        f"Expected each pickle chunk to be a list, got {type(chunk)}"
                    )

        return all_metrics
    
    @classmethod
    def load_all_run_metrics(cls, exp_dir):
        """
        Load all streamed per-run metrics files in sorted run order.
        Returns:
            list_of_runs where each entry is a flat list of episode metric dicts.
        """
        run_files = []
        for name in os.listdir(exp_dir):
            if name.startswith("metrics_run") and name.endswith(".pkl"):
                stem = name[len("metrics_run"):-len(".pkl")]
                if stem.isdigit():
                    run_files.append((int(stem), name))

        run_files.sort(key=lambda x: x[0])

        all_runs = []
        for run_idx, _ in run_files:
            all_runs.append(cls.load_run_metrics(exp_dir, run_idx))

        return all_runs
    
    @classmethod
    def load_agent_logs(cls, exp_dir):
        """
        Load agent_logs from all per-run streamed metric files.

        Returns:
            list[run][episode] = agent_logs
        """
        all_runs_metrics = cls.load_all_run_metrics(exp_dir)

        all_agent_logs = []
        for run in all_runs_metrics:
            episode_agent_logs = []
            for episode in run:
                episode_agent_logs.append(episode.get("agent_logs", None))
            all_agent_logs.append(episode_agent_logs)

        return all_agent_logs

    @classmethod
    def load_run_agent_logs(cls, exp_dir, run_idx):
        """
        Load agent_logs for a single run.

        Returns:
            list[episode] = agent_logs
        """
        run_metrics = cls.load_run_metrics(exp_dir, run_idx)

        episode_agent_logs = []
        for episode in run_metrics:
            episode_agent_logs.append(episode.get("agent_logs", None))

        return episode_agent_logs
    
    @staticmethod
    def concat_dicts_of_arrays(chunks, axis=0):
        """
        Concatenate a list of dicts whose values are numpy arrays.
        Unlike the old version, this uses the union of keys across chunks,
        so keys that appear only in later chunks are preserved.

        Returns:
        - None if chunks empty
        - dict[key] = np.concatenate([...], axis=axis)
        """
        if not chunks:
            return None

        all_keys = set()
        for c in chunks:
            all_keys.update(c.keys())

        out = {}
        for k in sorted(all_keys):
            arrs = []
            for c in chunks:
                if k not in c:
                    continue
                v = c[k]
                if not isinstance(v, np.ndarray):
                    raise TypeError(f"Value for key '{k}' must be np.ndarray, got {type(v)}")
                arrs.append(v)

            if len(arrs) == 0:
                continue

            out[k] = np.concatenate(arrs, axis=axis)

        return out
    
    def _metrics_stream_path(self, run_idx):
        return os.path.join(self.exp_dir, f"metrics_run{run_idx}.pkl")

    def _append_metrics_chunk(self, run_idx, metrics_chunk):
        """
        Append one chunk of episode metrics to the per-run pickle stream.
        Each chunk is itself a list of episode metric dicts.
        """
        if not metrics_chunk:
            return
        path = self._metrics_stream_path(run_idx)
        with open(path, "ab") as f:
            pickle.dump(metrics_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _reset_metrics_stream(self, run_idx):
        """
        Remove an existing streamed metrics file for this run so a fresh run
        does not append onto stale data.
        """
        path = self._metrics_stream_path(run_idx)
        if os.path.exists(path):
            os.remove(path)

    def _flush_metrics_buffer(self, run_idx, metrics_buffer):
        """
        Dump remaining buffered metrics to disk and clear the buffer in-place.
        """
        if self._dump_metrics and metrics_buffer:
            self._append_metrics_chunk(run_idx, metrics_buffer)
            metrics_buffer.clear()