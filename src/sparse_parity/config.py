"""Configuration constants for sparse parity experiments."""

from dataclasses import dataclass


@dataclass
class Config:
    """Experiment configuration. All fields have sensible defaults for 3-bit parity."""
    n_bits: int = 3
    k_sparse: int = 3
    n_train: int = 20
    n_test: int = 20
    hidden: int = 1000
    lr: float = 0.5
    wd: float = 0.01
    max_epochs: int = 10
    seed: int = 42
    patience: int = 10
    batch_size: int = 1

    @property
    def total_params(self):
        return self.hidden * self.n_bits + self.hidden + self.hidden + 1


# Preset for 20-bit scaling experiment
SCALE_CONFIG = Config(n_bits=20, k_sparse=3, n_train=200, n_test=200, hidden=2000, max_epochs=50)
