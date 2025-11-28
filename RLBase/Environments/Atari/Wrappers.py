import gymnasium as gym



class IdentityWrapper(gym.Wrapper):
    """No-op wrapper to keep the wrapper chain composable."""
    def __init__(self, env):
        super().__init__(env)


WRAPPING_TO_WRAPPER = {
    "Identity": IdentityWrapper,
}
