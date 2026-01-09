import torch

from ..utils import RandomGenerator
from typing import Any, Dict, Optional, Tuple, List, Sequence

class BaseOption(RandomGenerator):
    """
    Base class for a single option.

    Works for both:
      - Neural options (with networks, parameters on `device`)
      - Symbolic options (pure Python programs, no torch required)
    """

    def __init__(
        self,
        option_id: Optional[str] = None,
        hyper_params: Optional[Any] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.option_id = option_id  # e.g. "go_to_door", or index as string
        self.hp = hyper_params
        self.device = device

    # --------- Core interface ---------
    def can_initiate(self, observation: Any) -> bool:
        """
        Initiation set I_o(s) – by default always true.
        Override if your option has a restricted initiation set.
        """
        return True
    
    def should_initiate(self, observation: Any) -> bool:
        """
        It is a subset of the I_o that will suggest the agent to take this option
        """
        return False
    
    def reward_func(self, observation: Any) -> float:
        """
        Decide the intrinsic reward given to the agent related to the completeness of the option
        """
        raise NotImplementedError(
            "reward_func must be implemented by option subclasses"
        )

    def select_action(
        self,
        observation: Any,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Decide the *primitive action* given observation.

        Returns:
            action: env-compatible action (e.g. int for Discrete, np.array for Box)
        """
        raise NotImplementedError(
            "select_action must be implemented by option subclasses."
        )

    def is_terminated(
        self,
        observation: Any,
    ) -> bool:
        """
        Termination condition β_o(s). Called after each step.
        """
        raise NotImplementedError(
            "is_terminated must be implemented by option subclasses."
        )

    def reset(self, seed: Optional[int] = None):
        """
        Reset any internal state and RNG.
        """
        if seed is not None:
            self.set_seed(seed)
            

    # --------- Saving / Loading ---------

    def save(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        checkpoint = {
            "option_class": self.__class__.__name__,
            "option_id": self.option_id,
            "hyper_params": self.hp,
            "device": self.device,
            "rng_state": self.get_rng_state(),
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_option.t")
        return checkpoint

    @classmethod
    def load(cls, file_path: Optional[str] = None, checkpoint: Optional[Dict] = None):
        """
        Generic load that calls subclass _load_state_dict.
        Subclasses only need to override _state_dict/_load_state_dict.
        """
        if checkpoint is None:
            if file_path is None:
                raise ValueError("Either file_path or checkpoint must be provided.")
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        instance = cls(
            option_id=checkpoint.get("option_id", None),
            hyper_params=checkpoint.get("hyper_params", None),
            device=checkpoint.get("device", "cpu"),
        )
        instance.set_rng_state(checkpoint["rng_state"])
        return instance

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(id={self.option_id}, hp={self.hp})"
    
class BaseOfflineOptionLearner(RandomGenerator):
    """
    Base class for offline option discovery.

    Typical usage:
      - collect a dataset (transitions, trajectories, etc.)
      - feed it via update(...)
      - call learn() once to get a list of options
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hyper_params: Optional[Any] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hp = hyper_params
        self.device = device

    # ------------ Data ingestion ------------
    def update(self, *args, **kwargs):
        """
        Ingest data from agent / replay buffer.

        E.g. for an offline learner you might pass:
          update(transitions=batch_of_trajectories)
        or
          update(obs, action, reward, next_obs, done)
        Subclasses define the exact signature.
        """
        raise NotImplementedError

    # ------------ Learning ------------
    def learn(self) -> Sequence[BaseOption]:
        """
        Perform the actual option discovery and return the new option set.

        Returns:
            A sequence (list/tuple) of BaseOption instances.
        """
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.set_seed(seed)
        # Subclasses reset their internal buffers here.

    # ------------ Save / load ------------

    def save(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        checkpoint = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "hyper_params": self.hp,
            "device": self.device,
            "rng_state": self.get_rng_state(),
            "learner_class": self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_offline_option_learner.t")
        return checkpoint

    @classmethod
    def load(cls, file_path: Optional[str] = None, checkpoint: Optional[Dict] = None):
        if checkpoint is None:
            if file_path is None:
                raise ValueError("Either file_path or checkpoint must be provided.")
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        instance = cls(
            checkpoint["observation_space"],
            checkpoint["action_space"],
            hyper_params=checkpoint.get("hyper_params", None),
            device=checkpoint.get("device", "cpu"),
        )
        instance.set_rng_state(checkpoint["rng_state"])
        return instance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hp={self.hp})"
   
class BaseOnlineOptionLearner(RandomGenerator):
    """
    Base class for online / continual option learning.

    Intended to be called *during* RL training, e.g. every step / every N steps,
    potentially modifying the option set used by a policy.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hyper_params: Optional[Any] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hp = hyper_params
        self.device = device

    # ------------ Online updates ------------
    def update(self, *args, **kwargs):
        """
        Online update with incoming transitions.

        Example signature for a concrete implementation:
          update(obs, action, reward, next_obs, done, step_idx)

        This should *not* block for a long time; think 'small incremental step'.
        """
        raise NotImplementedError

    def get_options(self) -> Sequence[BaseOption]:
        """
        Return the current option set.
        Could be called by a BaseContinualPolicy during training.
        """
        raise NotImplementedError

    def trigger_option_learner(self):
        """
        Optional: explicitly trigger a (possibly heavy) re-learning or pruning
        step, separate from incremental update().
        """
        pass

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.set_seed(seed)
        # Subclasses clear internal buffers, counts, etc.

    # ------------ Save / load ------------

    def save(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        checkpoint = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "hyper_params": self.hp,
            "device": self.device,
            "rng_state": self.get_rng_state(),
            "learner_class": self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_online_option_learner.t")
        return checkpoint

    @classmethod
    def load(cls, file_path: Optional[str] = None, checkpoint: Optional[Dict] = None):
        if checkpoint is None:
            if file_path is None:
                raise ValueError("Either file_path or checkpoint must be provided.")
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)

        instance = cls(
            checkpoint["observation_space"],
            checkpoint["action_space"],
            hyper_params=checkpoint.get("hyper_params", None),
            device=checkpoint.get("device", "cpu"),
        )
        instance.set_rng_state(checkpoint["rng_state"])
        return instance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hp={self.hp})"