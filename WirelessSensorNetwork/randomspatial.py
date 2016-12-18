"""Random points in various topologies."""

import numpy as np


def on_sphere():
    """Uniform sample on unit sphere.unit

    Uses Muller method."""
    vec = np.random.standard_normal(3)
    return vec / np.linalg.norm(vec)


def in_disk():
    """Uniform sample in unit circle."""
    sample = np.random.random_sample(2)
    a, b = sample.sort()
    return np.array([b * np.cos(2 * np.pi * a / b),
                     b * np.sin(2 * np.pi * a / b)])


def in_square():
    """Uniform sample in unit square."""
    return np.random.random_sample(size=2)
