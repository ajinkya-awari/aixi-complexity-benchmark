"""Public observable-history key: the only state learners may condition on.

A key is the flattened last-m (action, observation) pairs followed by the current
observation. Rewards and latent states are not part of the key. Before m steps have
elapsed the window is left-padded with (-1, -1) pairs.
"""

PAD = (-1, -1)


def initial_window(m: int) -> tuple:
    return (PAD,) * m


def advance_window(window: tuple, action: int, obs: int, m: int) -> tuple:
    if m == 0:
        return ()
    return (window + ((action, obs),))[-m:]


def flat_key(window: tuple, obs: int) -> tuple:
    return tuple(x for pair in window for x in pair) + (obs,)


class KeyTracker:
    def __init__(self, m: int):
        self.m = m
        self.window = initial_window(m)
        self.obs = -1

    def reset(self, obs: int) -> tuple:
        self.window = initial_window(self.m)
        self.obs = obs
        return self.key()

    def key(self) -> tuple:
        return flat_key(self.window, self.obs)

    def step(self, action: int, obs: int) -> tuple:
        self.window = advance_window(self.window, action, self.obs, self.m)
        self.obs = obs
        return self.key()
