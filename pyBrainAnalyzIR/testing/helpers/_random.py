import numpy as np


def get_rng(random_seed=None):
    """Return a generator, reusing one passed through nested simulations."""
    if isinstance(random_seed, np.random.Generator):
        return random_seed
    return np.random.default_rng(random_seed)
