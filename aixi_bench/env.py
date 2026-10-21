"""Custom Gymnasium environment for a frozen finite-state task."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FiniteTaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(task.n_obs)
        self._cum_init = np.cumsum(task.init)
        self._cum_T = np.cumsum(task.T, axis=2)
        self._state = 0  # latent state: evaluator-side only, never exposed in info
        self._t = 0

    @staticmethod
    def _draw(cum, u):
        return min(int(np.searchsorted(cum, u, side="right")), len(cum) - 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self._state = self._draw(self._cum_init, self.np_random.random())
        self._t = 0
        return np.int64(self.task.obs_map[self._state]), {}

    def step(self, action):
        action = int(action)
        if action not in (0, 1):
            raise ValueError(f"invalid action {action}")
        s = self._state
        s2 = self._draw(self._cum_T[s, action], self.np_random.random())
        reward = float(self.task.R[s, action, s2])
        self._state = s2
        self._t += 1
        truncated = self._t >= self.task.horizon
        return np.int64(self.task.obs_map[s2]), reward, False, truncated, {}
