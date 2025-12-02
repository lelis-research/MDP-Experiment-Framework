"""
Hindsight Experience Replay (HER) buffer.

Stores full episodes and relabels goals when sampling.
Assumptions:
- Each transition is a dict with keys like: observation, achieved_goal, desired_goal, action, reward, next_observation, done.
- Relabeling strategies:
  * "future": replace desired_goal with a later achieved_goal from the same episode.
  * "final": replace desired_goal with the final achieved_goal of the episode.
Sampling:
- sample(batch_size) picks random episodes/timesteps, relabels each transition into k variants + the original, then returns one variant per sample.
Capacity:
- capacity limits total transitions; oldest episodes are dropped when over capacity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Literal, Sequence

from ..utils import RandomGenerator

Transition = Dict[str, Any]


class HERBuffer(RandomGenerator):
    def __init__(
        self,
        capacity: int,
        strategy: Literal["future", "final"] = "future",
        k: int = 4,
    ):
        """
        Args:
            capacity: max number of transitions to store (episodes will be truncated when full).
            strategy: goal relabeling strategy ("future" or "final").
            k: number of HER relabels per transition when sampling.
        """
        self.capacity = capacity
        self.strategy = strategy
        self.k = k
        self.episodes: List[List[Transition]] = []
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_episode(self, episode: List[Transition]) -> None:
        """Add a complete episode (list of transitions)."""
        if not episode:
            return
        self.episodes.append(episode)
        self.size += len(episode)
        self._trim()

    def _trim(self) -> None:
        while self.size > self.capacity and self.episodes:
            removed = self.episodes.pop(0)
            self.size -= len(removed)

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample transitions with HER relabeling. Returns a list of relabeled transitions.
        """
        if self.size == 0:
            raise ValueError("HERBuffer is empty")
        batch: List[Transition] = []
        for _ in range(batch_size):
            ep = self._rand_elem(self.episodes)
            t = self._rand_int(0, len(ep))
            transition = ep[t]
            her_transitions = self._relabel(transition, ep, t)
            batch.append(self._rand_elem(her_transitions))
        return batch

    def _relabel(self, transition: Transition, episode: List[Transition], t: int) -> List[Transition]:
        """
        Create k+1 transitions: the original and k HER relabeled variants.
        """
        variants = [transition]
        for _ in range(self.k):
            if self.strategy == "future":
                # pick a future step from the same episode
                if t + 1 >= len(episode):
                    future_idx = t
                else:
                    future_idx = self._rand_int(t + 1, len(episode))
                new_goal = episode[future_idx]["achieved_goal"]
            elif self.strategy == "final":
                new_goal = episode[-1]["achieved_goal"]
            else:
                raise ValueError(f"Unknown HER strategy: {self.strategy}")

            relabeled = dict(transition)
            relabeled["desired_goal"] = new_goal
            variants.append(relabeled)
        return variants


if __name__ == "__main__":
    # Create a toy episode with achieved goals 0..4
    episode = []
    for t in range(5):
        episode.append({
            "observation": t,
            "achieved_goal": t,
            "desired_goal": 10,  # dummy
            "action": t,
            "reward": 0,
            "next_observation": t + 1,
            "done": t == 4,
        })

    her = HERBuffer(capacity=20, strategy="future", k=2)
    her.add_episode(episode)
    batch = her.sample(3)
    print("Sampled relabeled transitions (desired_goal fields):", [b["desired_goal"] for b in batch])
