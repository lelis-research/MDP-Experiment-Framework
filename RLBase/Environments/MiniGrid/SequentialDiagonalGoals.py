import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
from minigrid.core.mission import MissionSpace


class SequentialDiagonalGoalsEnv(MiniGridEnv):
    """
    Big empty room. Goals appear one-by-one along a diagonal.
    Reaching the current goal gives reward, removes it, and spawns the next.
    Episode ends when the final (opposite-corner) goal is reached.

    Params
    ------
    size : int
        Outer grid size (with walls). Playable area is (size-2) x (size-2).
    main_diagonal : bool
        If True, goals go from top-left to bottom-right; if False, top-right to bottom-left.
    stride : int
        Step between consecutive goal cells along the diagonal (1 = every cell).
    agent_start_pos : (int,int) | None
        If None, start at the first diagonal cell (corner). Otherwise use provided.
    agent_start_dir : int
        0:right, 1:down, 2:left, 3:up (MiniGrid convention).
    per_goal_reward : float | None
        Reward for each intermediate goal. If None, we give 1/num_goals per goal so total reward ≈ 1.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "SequentialDiagonalGoals"}

    def __init__(
        self,
        size: int = 11,
        main_diagonal: bool = True,
        stride: int = 1,
        agent_start_pos=None,
        agent_start_dir: int = 0,
        per_goal_reward: float | None = None,
        **kwargs,
    ):
        assert size >= 5 and size % 2 == 1, "Use odd size >=5 for symmetry (e.g., 7, 9, 11)."
        assert stride >= 1
        self.room_size = size
        self.main_diagonal = main_diagonal
        self.stride = stride
        self.user_start_pos = agent_start_pos
        self.user_start_dir = agent_start_dir
        self.per_goal_reward = per_goal_reward

        # we’ll compute the sequence at reset (after grid exists)
        self.goal_positions: list[tuple[int, int]] = []
        self.goal_index: int = 0
        self.current_goal: Goal | None = None

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            agent_start_pos=None,  # we’ll set it in reset()
            agent_start_dir=agent_start_dir,
            max_steps=4 * size * size,  # generous
            **kwargs,
        )

    # ---------- required overrides ----------

    def _gen_mission(self):
        return "Collect goals along the diagonal in order until the opposite corner."

    def _gen_grid(self, width, height):
        # Create empty grid with outer walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Compute diagonal sequence inside the walls (interior is 1..w-2 / 1..h-2)
        interior_w, interior_h = width - 2, height - 2
        assert interior_w == interior_h, "Expect square interior."

        # Diagonal coordinates in the interior
        if self.main_diagonal:
            # from (1,1) to (w-2,h-2)
            diag = [(1 + i, 1 + i) for i in range(0, interior_w)]
        else:
            # from (w-2,1) to (1,h-2)
            diag = [(width - 2 - i, 1 + i) for i in range(0, interior_w)]

        # Apply stride and ensure we include both ends
        diag = [pos for idx, pos in enumerate(diag) if idx % self.stride == 0]
        if diag[-1] != (diag[0][0] + (len(diag) - 1) * (1 if self.main_diagonal else -1),
                        diag[0][1] + (len(diag) - 1)):
            # (safety; but above construction should already ensure endpoints)
            pass

        self.goal_positions = diag
        self.goal_index = 0

        # Agent start
        if self.user_start_pos is None:
            # Start at the first cell on the diagonal, facing right by default
            self.agent_pos = self.goal_positions[0]
            self.agent_dir = self.user_start_dir
        else:
            # respect provided position (ensure it’s inside)
            ax, ay = self.user_start_pos
            assert 1 <= ax <= width - 2 and 1 <= ay <= height - 2, "agent_start_pos must be inside walls"
            self.agent_pos = (ax, ay)
            self.agent_dir = self.user_start_dir

        # First goal should not be on top of the agent; if it is, skip it
        if self.agent_pos == self.goal_positions[0]:
            # If starting on the first goal, consider it immediately collected on first step()
            # (we’ll handle in step; here, place the next one if any)
            spawn_idx = 1 if len(self.goal_positions) > 1 else 0
        else:
            spawn_idx = 0

        # Place the initial visible goal (if any remain)
        self._spawn_goal(spawn_idx)

        self.mission = self._gen_mission()

    # ---------- helpers ----------

    def _spawn_goal(self, idx: int):
        # Clear previous
        if self.current_goal is not None:
            # remove from grid
            gx, gy = self.current_goal.cur_pos
            self.grid.set(gx, gy, None)
            self.current_goal = None

        self.goal_index = idx
        if idx < len(self.goal_positions):
            gx, gy = self.goal_positions[idx]
            # Don’t place on the agent; if same cell, we’ll immediately count it on step()
            self.current_goal = Goal()
            self.place_obj(self.current_goal, top=(gx, gy), size=(1, 1), max_tries=1)

    def _on_goal_collected(self):
        # Compute reward
        if self.per_goal_reward is None:
            # normalize so total ~1 across all goals (including the last)
            r = 1.0 / len(self.goal_positions)
        else:
            r = float(self.per_goal_reward)
        return r

    # ---------- main logic ----------

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # If episode hasn’t ended by default logic, check our sequential-goal mechanics
        if not terminated and not truncated:
            # Did we land on the current goal?
            if self.current_goal is not None:
                if tuple(self.agent_pos) == tuple(self.goal_positions[self.goal_index]):
                    # Collect
                    reward += self._on_goal_collected()

                    next_idx = self.goal_index + 1
                    if next_idx >= len(self.goal_positions):
                        # final goal reached
                        terminated = True
                        # remove the last goal for cleanliness
                        self._spawn_goal(next_idx)  # clears previous
                    else:
                        # spawn next goal
                        self._spawn_goal(next_idx)

        return obs, reward, terminated, truncated, info