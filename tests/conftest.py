import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pytest
import torch


@pytest.fixture
def tiny_graph():
    """Two triangles joined by a bridge plus one isolated node (node 6)."""
    und = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    ei = np.array([[a for a, b in und] + [b for a, b in und], [b for a, b in und] + [a for a, b in und]])
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 5)).astype(np.float32)
    return x, ei, 7
