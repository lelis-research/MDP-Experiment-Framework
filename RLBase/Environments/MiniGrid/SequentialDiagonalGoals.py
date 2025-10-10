from typing import Any
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
from minigrid.core.mission import MissionSpace
from gymnasium.envs.registration import register


class SequentialDiagonalGoalsEnv(MiniGridEnv):
    """
    Big empty room. A single visible Goal appears at a time along a diagonal.
    When the agent reaches the current goal, it disappears, reward is given,
    and the next goal spawns. Episode ends after the final goal (opposite corner).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10, "name": "SequentialDiagonalGoals"}

    def __init__(
        self,
        *,
        grid_size: int = 11,
        main_diagonal: bool = True,
        stride: int = 1,
        per_goal_reward: float | None = None,
        **kwargs: Any,
    ):
        self.main_diagonal = main_diagonal
        self.stride = stride
        self.per_goal_reward = per_goal_reward

        # sequential-goal state (filled in _gen_grid)
        self.goal_positions: list[tuple[int, int]] = []
        self.goal_index: int = 0
        self.current_goal: Goal | None = None

        mission_space = MissionSpace(
            mission_func=lambda: "Collect goals along the diagonal in order until the opposite corner."
        )

        # follow parent signature; don't pass agent_start_* (not supported here)
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            see_through_walls=True,
            **kwargs,  # may include max_steps, render_mode, etc.
        )

    # --- required by MiniGridEnv; must set grid, agent_pos, agent_dir ---
    def _gen_grid(self, width: int, height: int):
        # empty grid with outer walls already created in MiniGridEnv.__init__,
        # but we replace to be explicit and fresh per episode:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        n = width - 2
        assert n == height - 2, "Use a square grid_size for a diagonal path."

        # build diagonal coordinates inside the walls
        if self.main_diagonal:
            diag = [(1 + i, 1 + i) for i in range(n)]
        else:
            diag = [(width - 2 - i, 1 + i) for i in range(n)]

        # sub-sample by stride (1 = every cell)
        self.goal_positions = [p for i, p in enumerate(diag) if i % self.stride == 0]
        assert len(self.goal_positions) >= 1, "No goal cells; check stride/grid_size."

        # place agent at the first diagonal cell, facing right (0)
        self.agent_pos = self.goal_positions[0]
        self.agent_dir = 0  # 0:right, 1:down, 2:left, 3:up

        # spawn first visible goal; if agent starts on first, spawn the next one
        start_on_first = True  # by design above
        spawn_idx = 1 if (start_on_first and len(self.goal_positions) > 1) else 0
        self._spawn_goal(spawn_idx)

    def _spawn_goal(self, idx: int):
        # Safely clear previous goal (only if it was actually placed)
        if self.current_goal is not None:
            prev_pos = getattr(self.current_goal, "cur_pos", None)
            if prev_pos is not None:
                px, py = prev_pos
                self.grid.set(px, py, None)
            self.current_goal = None

        self.goal_index = idx

        # If we've exhausted the sequence, nothing to spawn
        if idx >= len(self.goal_positions):
            return

        gx, gy = self.goal_positions[idx]

        # Never place a goal on the agent's cell
        if tuple(self.agent_pos) == (gx, gy):
            # If this ever happens, skip forward (rare; but safe)
            next_idx = idx + 1
            self._spawn_goal(next_idx)
            return

        # Deterministic placement: set into the exact cell
        goal = Goal()
        self.grid.set(gx, gy, goal)
        # (Grid.set already sets init_pos/cur_pos, but the next 2 lines keep us robust
        # across older forks/versions that may not do it)
        goal.init_pos = (gx, gy)
        goal.cur_pos = (gx, gy)
        self.current_goal = goal

    def _per_goal_reward(self) -> float:
        # normalize so total ≈ 1 by default
        return (1.0 / len(self.goal_positions)) if self.per_goal_reward is None else float(self.per_goal_reward)

    # --- custom step: replicate MiniGrid step semantics but alter goal handling ---
    def step(self, action):
        # this mirrors MiniGridEnv.step, except goal handling is sequential (no auto-terminate on first goal)
        self.step_count += 1

        reward = 0.0
        terminated = False
        truncated = False

        # compute forward interactions
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # actions
        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4

        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)

            # If we landed on the active goal cell, collect and spawn the next
            if (
                self.current_goal is not None
                and tuple(self.agent_pos) == self.goal_positions[self.goal_index]
            ):
                reward += (1.0 / len(self.goal_positions)) if self.per_goal_reward is None else float(self.per_goal_reward)
                next_idx = self.goal_index + 1
                if next_idx >= len(self.goal_positions):
                    terminated = True
                    # clears previous goal; nothing new to place
                    self._spawn_goal(next_idx)
                else:
                    self._spawn_goal(next_idx)

        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup() and self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = (-1, -1)
                self.grid.set(fwd_pos[0], fwd_pos[1], None)

        elif action == self.actions.drop:
            if self.grid.get(*fwd_pos) is None and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}
    
register(
    id='SequentialDiagonalGoalsEnv-v0',
    entry_point=SequentialDiagonalGoalsEnv
)