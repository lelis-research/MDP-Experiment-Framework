from tqdm import tqdm
import os
import numpy as np
import torch  # used only for best-agent checkpointing passthrough
import random

from . import BaseExperiment


class ParallelExperiment(BaseExperiment):
    """
    Parallel/vectorized variant of BaseExperiment.
    Overrides only the run loops; everything else (multi_run, _one_run, etc.)
    is reused from BaseExperiment.
    Assumptions:
      - self.env is a vectorized env with attribute `num_envs`
      - step() → (obs, rewards, terminateds, truncateds, infos) with per-env arrays
      - agent exposes parallel_act(...) and parallel_update(...)
    """
    def _extract_actual_rewards(self, infos, rewards):
        if isinstance(infos, (list, tuple)) and len(infos) == len(rewards):
            return np.asarray(
                [info.get("actual_reward", rewards[j]) if isinstance(info, dict) else rewards[j]
                for j, info in enumerate(infos)],
                dtype=np.float32,
            )
        if isinstance(infos, dict) and "actual_reward" in infos:
            ar = np.asarray(infos["actual_reward"], dtype=np.float32)
            if ar.shape == rewards.shape:
                return ar
        return rewards

    def _obs_i(self, obs, i):
        if isinstance(obs, dict):
            return {k: (v[i] if hasattr(v, "__getitem__") else v) for k, v in obs.items()}
        return obs[i]

    def _single_run_episodes(self, env, agent, num_episodes, seed, run_idx):
        """
        Vectorized episodes runner: completes `num_episodes` total across all sub-envs.
        Returns: (all_metrics, best_agent_checkpoint_dict)
        """
        best_agent, best_return = None, -np.inf
        if self._train:
            agent.reset(seed)
        
        all_metrics = []
        pbar = tqdm(total=num_episodes, desc="Running episodes")
        
        num_envs = env.num_envs
        
        # # Per-sub-env trackers
        ep_returns = np.zeros(num_envs, dtype=np.float32)
        ep_lengths = np.zeros(num_envs, dtype=np.int64)
        transitions_buf = [[] for _ in range(num_envs)] if self._dump_transitions else None

        episodes_done = 0

        observations, infos = env.reset()
        while episodes_done < num_episodes:                
            actions = agent.parallel_act(observations)
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
            
            if self._dump_transitions:
                for i in range(num_envs):
                    transitions_buf[i].append((self._obs_i(observations, i), actions[i], rewards[i],
                                               bool(terminateds[i]), bool(truncateds[i])))
            
            
            ep_returns += self._extract_actual_rewards(infos, rewards)
            ep_lengths += 1

            if self._train:
                agent.parallel_update(next_observations, rewards, terminateds, truncateds)
                
            observations = next_observations

            dones = np.logical_or(terminateds, truncateds)
            # Record finished episodes (may be multiple per step)
            for i in range(len(dones)):
                if not dones[i]:
                    continue
                episodes_done += 1
                                
                try:
                    frames = env.envs[i].render()
                except:
                    frames = []
                
    
                metrics = {
                    "ep_return": float(ep_returns[i]),
                    "ep_length": int(ep_lengths[i]),
                    "frames": frames,
                    "transitions": transitions_buf[i] if self._dump_transitions else [],
                    "agent_seed": seed,
                    "episode_index": episodes_done,
                }
                all_metrics.append(metrics)

                # Best agent snapshot (checkpoint dict from agent.save())
                if ep_returns[i] >= best_return:
                    best_return = ep_returns[i]
                    best_agent = agent.save()

                # Progress + optional checkpoint
                pbar.update(1)
                pbar.set_postfix({"Return": metrics["ep_return"], "Steps": metrics["ep_length"]})

                if self._checkpoint_freq is not None and self._checkpoint_freq != 0 and episodes_done % self._checkpoint_freq == 0:
                    path = os.path.join(self.exp_dir, f"Run{run_idx}_E{episodes_done}")
                    agent.save(path)

                # Reset per-sub-env trackers for that env slot
                ep_returns[i] = 0.0
                ep_lengths[i] = 0
                if self._dump_transitions:
                    transitions_buf[i] = []
            

        pbar.close()
        return all_metrics, best_agent

    def _single_run_steps(self, env, agent, total_steps, seed, run_idx):
        """
        Vectorized steps runner: executes ~`total_steps` interactions in aggregate.
        Each call to env.step() counts as `num_envs` interactions.
        If the limit is hit mid-episode, we truncate and emit partial episodes.
        Returns: (all_metrics, best_agent_checkpoint_dict)
        """
        best_agent, best_return = None, -np.inf
        if self._train:
            agent.reset(seed)

        all_metrics = []
        pbar = tqdm(total=total_steps, desc="Running steps")
        
        num_envs = env.num_envs

        ep_returns = np.zeros(num_envs, dtype=np.float32)
        ep_lengths = np.zeros(num_envs, dtype=np.int64)
        transitions_buf = [[] for _ in range(num_envs)] if self._dump_transitions else None

        steps_so_far = 0
        episodes_done = 0
        
        observations, infos = env.reset()
        while steps_so_far < total_steps:
            actions = agent.parallel_act(observations)
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
            
            if self._dump_transitions:
                for i in range(num_envs):
                    transitions_buf[i].append((self._obs_i(observations, i), actions[i], rewards[i],
                                               bool(terminateds[i]), bool(truncateds[i])))
            
                 
            ep_returns += self._extract_actual_rewards(infos, rewards)
            ep_lengths += 1
            steps_so_far += num_envs  # aggregate interactions across sub-envs
            
            if steps_so_far >= total_steps:
                truncateds = np.ones_like(truncateds)
            
            if self._train:
                agent.parallel_update(next_observations, rewards, terminateds, truncateds)

            pbar.update(num_envs)
            observations = next_observations
            

            # Determine which envs finished this step (natural or forced)
            dones = np.logical_or(terminateds, truncateds)


            # Emit metrics for all finished slots
            for i in range(len(dones)):
                if not dones[i]:
                    continue
                episodes_done += 1
                
                try:
                    frames = env.envs[i].render()
                except:
                    frames = []

                metrics = {
                    "ep_return": float(ep_returns[i]),
                    "ep_length": int(ep_lengths[i]),
                    "frames": frames,
                    "transitions": transitions_buf[i] if self._dump_transitions else [],
                    "agent_seed": seed,
                    "episode_index": episodes_done,
                }
                all_metrics.append(metrics)
                
                # Best agent snapshot (checkpoint dict from agent.save())
                if ep_returns[i] >= best_return:
                    best_return = ep_returns[i]
                    best_agent = agent.save()

                
                pbar.set_postfix({"Episode": episodes_done, 
                                  "Return": metrics["ep_return"], 
                                  "TotalSteps": steps_so_far})
                
                if self._checkpoint_freq is not None and self._checkpoint_freq != 0 and episodes_done % self._checkpoint_freq == 0:
                    path = os.path.join(self.exp_dir, f"Run{run_idx}_E{episodes_done}")
                    agent.save(path)

                # Reset per-slot accumulators
                ep_returns[i] = 0.0
                ep_lengths[i] = 0
                if self._dump_transitions:
                    transitions_buf[i] = []


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
        
        self.env, self.agent = env, agent
        if tuning_hp is not None:
            agent.set_hp(tuning_hp)

        if case_num == 1:
            result, best_agent = self._single_run_episodes(env, agent, num_episodes, seed, run_idx)
        else:
            result, best_agent = self._single_run_steps(env, agent, total_steps, seed, run_idx)
                
        
        # Save last and best agent
        if self._checkpoint_freq is not None:
            path = os.path.join(self.exp_dir, f"Run{run_idx}_Last")
            agent.save(path)
            
            path = os.path.join(self.exp_dir, f"Run{run_idx}_Best")
            torch.save(best_agent, f"{path}_agent.t")
        
        return result