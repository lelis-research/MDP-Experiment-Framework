from __future__ import annotations
from typing import Any, Dict, List, Tuple
from gymnasium.envs.registration import register
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Door, Key, Goal, Wall


# ---------- Pickle-safe mission text ----------
def phased_option_mission() -> str:
    return "Solve the phase-specific key/door/switch puzzle to reach the goal."


# ---------- Custom World Objects ----------
class Switch(WorldObj):
    """
    A simple switch that toggles linked doors' open/closed state.
    - color is cosmetic
    - id_str used for stable IDs across phases
    """

    def __init__(self, color: str = "yellow", id_str: str = "S1", linked_doors: List[Door] | None = None):
        super().__init__("switch", color)
        self.id_str = id_str
        self.linked_doors: List[Door] = linked_doors or []
        self.is_on = False  # purely informational

    def can_overlap(self) -> bool:
        return True

    def toggle(self, env, pos):
        # flip hardware state
        self.is_on = not self.is_on
        # toggle all linked doors
        for d in self.linked_doors:
            d.is_open = not d.is_open
        # event hook
        if hasattr(env, "report_event"):
            env.report_event(("toggle_switch", self.id_str))


# ---------- Phase specs (edit freely) ----------
# All coordinates are absolute grid cells (0..width-1).
# We assume grid_size >= 11 for clarity; adjust if you shrink.
PHASE_SPECS: Dict[int, Dict[str, Any]] = {
    # Phase 1: Red key -> Red door bottleneck -> Goal
    1: dict(
        red_key_pos=(2, 2),
        red_door_pos=(5, 5),
        blue_key_pos=None,
        blue_door_pos=None,
        switch_pos=None,
        goal_pos=(8, 5),
        color_remap=None,  # no remap
        walls=[("vert", 5, 1, 9)],  # a vertical wall at x=5 from y=1..8
        door_links={},  # no switch links yet
    ),
    # Phase 2: Add blue door controlled by a switch; goal moves further
    2: dict(
        red_key_pos=(2, 2),
        red_door_pos=(5, 5),
        blue_key_pos=None,
        blue_door_pos=(7, 5),        # new door
        switch_pos=(3, 5),           # switch toggles the blue door
        goal_pos=(9, 5),
        color_remap=None,
        walls=[("vert", 5, 1, 9), ("vert", 7, 1, 9)],
        door_links={"S1": ["blue"]},  # switch S1 toggles blue door
    ),
    # Phase 3: Remap colors and cross-link switch
    3: dict(
        red_key_pos=(2, 2),
        red_door_pos=(7, 5),         # doors swap positions/colors
        blue_key_pos=None,
        blue_door_pos=(5, 5),
        switch_pos=(3, 5),
        goal_pos=(9, 5),
        color_remap={"red": "blue", "blue": "red"},  # test option re-binding
        walls=[("vert", 5, 1, 9), ("vert", 7, 1, 9)],
        door_links={"S1": ["red"]},  # now switch toggles the *other* door
    ),
}


def _color_after_remap(color: str, remap: Dict[str, str] | None) -> str:
    if remap and color in remap:
        return remap[color]
    return color


class PhasedOptionEnv(MiniGridEnv):
    """
    A phased MiniGrid environment to test continual addition of options.
    Subgoals: pick key(color), open door(color), toggle switch(id), reach goal.

    Exposed helpers:
      - set_phase(phase: int)
      - next_phase()
      - report_event(event_tuple)
      - semantic_state() -> dict
      - events (list of tuples)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10, "name": "PhasedOptionEnv"}

    def __init__(
        self,
        *,
        grid_size: int = 11,
        start_phase: int = 1,
        per_event_reward: float | None = None,
        **kwargs: Any,
    ):
        # high-level config
        self.grid_size_config = grid_size
        self.phase = start_phase
        self.per_event_reward = per_event_reward  # if None, normalized

        # runtime state
        self.events: List[Tuple] = []
        self.doors_by_color: Dict[str, Door] = {}
        self.switches_by_id: Dict[str, Switch] = {}
        self.goal_obj: Goal | None = None
        self._num_expected_events = 0  # used for reward normalization per phase

        # mission space (pickle-safe)
        mission_space = MissionSpace(mission_func=phased_option_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            see_through_walls=True,
            **kwargs,  # may include max_steps, render_mode, etc.
        )

    # ---------- Phase control ----------
    def set_phase(self, phase: int):
        assert phase in PHASE_SPECS, f"Unknown phase: {phase}"
        self.phase = phase

    def next_phase(self):
        nxt = self.phase + 1
        if nxt in PHASE_SPECS:
            self.phase = nxt

    # ---------- Event & state helpers ----------
    def report_event(self, event: Tuple):
        """Log a subgoal event (used by Switch, doors, key pickup, goal)."""
        self.events.append(event)

    def semantic_state(self) -> Dict[str, Any]:
        """Compact, symbolic snapshot for options (termination/initiation)."""
        return dict(
            phase=self.phase,
            carrying=None if self.carrying is None else self.carrying.type,
            carrying_color=None if self.carrying is None else self.carrying.color,
            doors={c: d.is_open for c, d in self.doors_by_color.items()},
            switches={sid: sw.is_on for sid, sw in self.switches_by_id.items()},
            agent_pos=tuple(self.agent_pos),
            goal_pos=None if self.goal_obj is None else tuple(self.goal_obj.cur_pos),
        )

    # ---------- Grid generation ----------
    def _gen_grid(self, width: int, height: int):
        spec = PHASE_SPECS[self.phase]
        self.events = []               # clear on new episode
        self.doors_by_color = {}
        self.switches_by_id = {}
        self.goal_obj = None

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Internal walls
        for kind, x_or_y, y1, y2 in spec.get("walls", []):
            if kind == "vert":
                x = x_or_y
                for y in range(y1, y2 + 1):
                    self.grid.set(x, y, Wall())
            elif kind == "horz":
                y = x_or_y
                for x in range(y1, y2 + 1):
                    self.grid.set(x, y, Wall())

        # Apply optional color remap
        remap = spec.get("color_remap", None)

        # Doors
        red_door_pos = spec.get("red_door_pos", None)
        blue_door_pos = spec.get("blue_door_pos", None)

        if red_door_pos:
            red_col = _color_after_remap("red", remap)
            d = Door(red_col, is_open=False, is_locked=False)
            self.grid.set(*red_door_pos, d)
            self.doors_by_color["red"] = d  # keep key by *logical* color (pre-remap label)

        if blue_door_pos:
            blue_col = _color_after_remap("blue", remap)
            d = Door(blue_col, is_open=False, is_locked=False)
            self.grid.set(*blue_door_pos, d)
            self.doors_by_color["blue"] = d

        # Keys (optional blue key; typically unnecessary for unlocked doors but useful if you switch to locked=True)
        red_key_pos = spec.get("red_key_pos", None)
        blue_key_pos = spec.get("blue_key_pos", None)

        if red_key_pos:
            red_col = _color_after_remap("red", remap)
            self.grid.set(*red_key_pos, Key(red_col))
        if blue_key_pos:
            blue_col = _color_after_remap("blue", remap)
            self.grid.set(*blue_key_pos, Key(blue_col))

        # Switches & links
        door_links = spec.get("door_links", {})
        if spec.get("switch_pos", None):
            sp = spec["switch_pos"]
            sw = Switch(color="yellow", id_str="S1", linked_doors=[])
            self.grid.set(*sp, sw)
            self.switches_by_id["S1"] = sw

            # link logical colors to actual Door instances
            targets = door_links.get("S1", [])
            for logical_color in targets:
                d = self.doors_by_color.get(logical_color)
                if d is not None:
                    sw.linked_doors.append(d)

        # Goal
        goal_pos = spec.get("goal_pos", (width - 2, height // 2))
        g = Goal()
        self.grid.set(*goal_pos, g)
        self.goal_obj = g

        # Agent start (left side, mid-height, empty cell)
        # Ensure not on a wall/door/switch
        start = (1, height // 2)
        if self.grid.get(*start) is not None:
            # fallback a bit to find a free cell
            for dx in range(1, 4):
                cand = (1 + dx, height // 2)
                if self.grid.get(*cand) is None:
                    start = cand
                    break
        self.agent_pos = start
        self.agent_dir = 0  # face right

        # Reward normalization target per phase
        # Define expected events per phase (tune to your liking)
        expected = 1  # reaching goal
        if red_key_pos:
            expected += 1
        if red_door_pos:
            expected += 1
        if blue_door_pos:
            expected += 1
        if spec.get("switch_pos", None):
            expected += 1
        self._num_expected_events = expected

    # ---------- Reward helpers ----------
    def _per_event_reward(self) -> float:
        if self.per_event_reward is not None:
            return float(self.per_event_reward)
        # Normalize so total ~ 1 over typical sequence of events
        return 1.0 / max(self._num_expected_events, 1)

    # ---------- Main step override (adds subgoal events & shaped rewards) ----------
    def step(self, action):
        # Keep parent dynamics (movement, pickup, toggle, door handling, termination on lava if any)
        obs, reward, terminated, truncated, info = super().step(action)

        # Subgoal: pickup key
        if action == self.actions.pickup:
            if self.carrying is not None and self.carrying.type == "key":
                # Report in *logical* color space (before remap) is tricky; we just log the actual color
                self.report_event(("pickup_key", self.carrying.color))
                reward += self._per_event_reward()

        # Subgoal: opening a door (detect just-turned-open under/adjacent)
        # We check the agent's front cell after forward moves; here we'll scan local door states and reward on first-time open
        # (In practice, you might keep a 'door_opened_once' set to avoid double-counting)
        if action in (self.actions.forward, self.actions.toggle):
            for logical_color, door in self.doors_by_color.items():
                # reward only at the transition to open
                if getattr(door, "_was_open", False) is False and door.is_open:
                    self.report_event(("open_door", logical_color))
                    reward += self._per_event_reward()
                door._was_open = door.is_open  # track

        # Subgoal: toggling a switch
        if action == self.actions.toggle:
            # The Switch object itself calls env.report_event(("toggle_switch", id))
            # Here we could add reward if the event was just added (we'll check tail of events list)
            if len(self.events) > 0 and self.events[-1][0] == "toggle_switch":
                reward += self._per_event_reward()

        # Terminal: reaching the goal (parent already terminates on stepping onto Goal)
        if terminated:
            # make sure we log reaching goal exactly once
            if len(self.events) == 0 or self.events[-1][0] != "reach_goal":
                self.report_event(("reach_goal",))
                reward += self._per_event_reward()

        return obs, reward, terminated, truncated, info
    
    

register(
    id='PhasedOptionEnv-v0',
    entry_point=PhasedOptionEnv
)