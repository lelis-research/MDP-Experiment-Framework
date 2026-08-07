import re
import numpy as np
from gymnasium.spaces import Discrete

from ..Base import BaseAgent
from ..Utils.HelperFunctions import get_single_observation_nobatch
from ...registry import register_agent
from .Utils import VulcanLLM


CRAFTER_SYSTEM_PROMPT = """You are an agent playing Crafter, a 2D survival game with a technology tree of 22 achievements.

GOAL: Unlock as many achievements as possible by surviving and progressing through the tech tree.

TECH TREE (must follow dependencies):
- collect_wood → place_table → make_wood_pickaxe → make_wood_sword
- make_wood_pickaxe → collect_stone → place_stone → make_stone_pickaxe → make_stone_sword
- make_stone_pickaxe → collect_coal + collect_iron → place_furnace → make_iron_pickaxe → make_iron_sword
- collect_diamond (requires iron pickaxe)
- collect_sapling → place_plant (food source)
- eat_cow, eat_plant (restore food)
- drink_water (restore drink by standing near water and drinking)
- defeat_zombie, defeat_skeleton (combat)

SURVIVAL: Keep health, food, drink, and energy above 0. Sleep (do_nothing near flat ground at night) to restore energy.

OUTPUT FORMAT: First reason briefly about your situation and next goal, then output your chosen action on the last line as:
ACTION: <action_name>"""


def _build_prompt(obs_text, inventory, status, available_actions, history):
    history_str = "\n".join([f"  Step {i+1}: {a}" for i, a in enumerate(history[-5:])])
    actions_str = ", ".join(available_actions)
    return f"""=== CURRENT STATE ===
Status: {status}
Inventory: {inventory}
Observation: {obs_text}

=== RECENT ACTIONS (last 5) ===
{history_str if history else "  None"}

=== AVAILABLE ACTIONS ===
{actions_str}

What do you do next?"""


@register_agent
class LLMAgent(BaseAgent):
    """
    LLM-based agent that selects actions by querying a language model.

    Expected hyper_params:
        api_key       (str)  : API key for the LLM endpoint.
        model         (str)  : Model name, e.g. "qwen3-235b".
        system_prompt (str)  : System prompt (defaults to CRAFTER_SYSTEM_PROMPT).
        max_tokens    (int)  : Max tokens for LLM response (default 512).
        temperature   (float): Sampling temperature (default 0.0).
    """
    name = "LLM"
    SUPPORTED_ACTION_SPACES = (Discrete,)

    def __init__(self, action_space, observation_space, hyper_params,
                 num_envs, feature_extractor_class, device="cpu"):
        super().__init__(action_space, observation_space, hyper_params,
                         num_envs, feature_extractor_class, device=device)

        self.llm = VulcanLLM(
            model=getattr(hyper_params, "model", "qwen3-235b"),
            api_key=getattr(hyper_params, "api_key", None),
        )

        # Per-env action history (list of action name strings)
        self._history = [[] for _ in range(num_envs)]

    # ------------------------------------------------------------------
    # ACTION SELECTION
    # ------------------------------------------------------------------
    def act(self, observation):
        actions = []
        for i in range(self.num_envs):
            obs_i = get_single_observation_nobatch(observation, i)

            # Extract structured fields from obs dict
            status, inventory_str, surroundings, available_actions = \
                self._extract_obs_fields(obs_i)

            prompt = _build_prompt(
                obs_text=surroundings,
                inventory=inventory_str,
                status=status,
                available_actions=available_actions,
                history=self._history[i],
            )

            system = getattr(self.hp, "system_prompt", None) or CRAFTER_SYSTEM_PROMPT
            max_tokens = getattr(self.hp, "max_tokens", 512)
            temperature = getattr(self.hp, "temperature", 0.0)

            response = self.llm(prompt, system=system,
                                max_tokens=max_tokens, temperature=temperature)
            action, action_name = self._parse_action(response, available_actions)
            self._history[i].append(action_name)
            actions.append(action)

        return np.array(actions, dtype=np.int64)

    # ------------------------------------------------------------------
    # OBS EXTRACTION
    # ------------------------------------------------------------------
    def _extract_obs_fields(self, obs):
        """Return (status, inventory_str, surroundings, available_actions_list)."""
        if isinstance(obs, dict):
            status       = str(obs.get("status", ""))
            inventory_str = str(obs.get("inventory_str", ""))
            surroundings = str(obs.get("surroundings", obs.get("obs", str(obs))))
            avail_raw    = obs.get("available_actions", "")
            if avail_raw:
                available_actions = [a.strip() for a in str(avail_raw).split(",") if a.strip()]
            else:
                available_actions = [str(i) for i in range(self.action_space.n)]
            # Fall back to full obs if structured fields are empty
            if not surroundings and "obs" in obs:
                surroundings = str(obs["obs"])
        else:
            status = inventory_str = ""
            surroundings = str(obs)
            available_actions = [str(i) for i in range(self.action_space.n)]
        return status, inventory_str, surroundings, available_actions

    # ------------------------------------------------------------------
    # RESPONSE PARSING
    # ------------------------------------------------------------------
    def _parse_action(self, response, available_actions):
        """Return (action_index, action_name). Tries ACTION: <name> first, then integer."""
        # Try "ACTION: <name>" format
        match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
        if match:
            name = match.group(1).lower()
            for idx, an in enumerate(available_actions):
                if an.lower() == name:
                    return idx, an

        # Fall back to bare integer
        match = re.search(r"\d+", response.strip())
        if match:
            action = int(match.group())
            action = max(0, min(action, self.action_space.n - 1))
            name = available_actions[action] if action < len(available_actions) else str(action)
            return action, name

        # Random fallback
        action = int(self._rand_int(0, self.action_space.n))
        name = available_actions[action] if action < len(available_actions) else str(action)
        return action, name
