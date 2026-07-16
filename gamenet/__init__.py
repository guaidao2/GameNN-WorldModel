"""
GameNN World Model — A standalone decision architecture with
structured state representation, recurrent decision steps,
hierarchical strategy/action selection, and one-step world model imagination.

Core Components
---------------
- GameNNConfig    — Fully configurable hyperparameters for any domain
- GameNNModel     — Complete decision architecture (the backbone)
- StateEncoder    — Observation → structured latent state
- RNNDecisionStep — GRU-style recurrent decision step
- ActionValueHead — Strategy/action/value heads
- WorldModelStep  — One-step RSSM-inspired outcome prediction
- Fuser           — Decision-to-output fusion
- GumbelRouter    — (Optional) Gumbel-Softmax discrete choice
"""

from .config import GameNNConfig
from .core import (
    StateEncoder,
    GumbelRouter,
    RNNDecisionStep,
    ActionValueHead,
    WorldModelStep,
    Fuser,
)
from .model import GameNNModel

__all__ = [
    # Config
    "GameNNConfig",
    # Main model
    "GameNNModel",
    # Core modules
    "StateEncoder",
    "GumbelRouter",
    "RNNDecisionStep",
    "ActionValueHead",
    "WorldModelStep",
    "Fuser",
]
