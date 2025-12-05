from __future__ import annotations

from typing import Iterable, List, Optional, Type

from ...Base import BaseOfflineOptionLearner, BaseOption


class RandomLearner(BaseOfflineOptionLearner):
    """
    Offline learner that randomly samples from a provided set of option classes.
    Intended to be passed an initial list of option classes at construction.
    """

    name = "RandomLearner"

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        hyper_params=None,
        device: str = "cpu",
        option_classes: Optional[Iterable[Type[BaseOption]]] = None,
    ):
        super().__init__(observation_space, action_space, hyper_params=hyper_params, device=device)
        self.option_classes: List[Type[BaseOption]] = list(option_classes) if option_classes else []

    def update(self, *args, **kwargs):
        # No dataset needed; selection is purely random at learn time.
        return None

    def learn(self):
        """
        Return a random subset (could be the full set) of the provided option classes.
        Hyper-parameters:
          - num_options: how many options to sample (defaults to all).
          - option_len: length passed to each option constructor (if supported).
        """
        if not self.option_classes:
            return []

        num_options = getattr(self.hp, "num_options", None) if self.hp is not None else None
        option_len = getattr(self.hp, "option_len", 20) if self.hp is not None else 20

        if num_options is None or num_options >= len(self.option_classes):
            chosen = self.option_classes
        else:
            chosen = self._rand_subset(self.option_classes, num_options)

        # Instantiate with option_len where applicable; fall back to default ctor.
        options = []
        for cls in chosen:
            try:
                opt = cls(option_len=option_len)
            except TypeError:
                opt = cls()
            options.append(opt)

        return options
