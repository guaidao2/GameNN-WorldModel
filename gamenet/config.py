"""
GameNN World Model — Configuration.

Defines the structural hyperparameters for the entire architecture.
All domain-specific labels (strategy names, action names) are configurable,
making GameNN applicable to any decision domain.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GameNNConfig:
    """
    Configuration for the GameNN World Model architecture.

    The architecture is fully domain-agnostic: strategy/action names,
    state dimensions, and all hyperparameters are configurable here.

    Example
    -------
    >>> config = GameNNConfig(
    ...     state_dim=16,
    ...     n_strategies=3,
    ...     n_actions=8,
    ...     hidden_dim=768,
    ... )
    """
    # ── Observation / Encoding ─────────────────────────────────────
    hidden_dim: int = 768
    """Dimensionality of the input observation vector (e.g. backbone hidden)."""

    state_dim: int = 16
    """Dimensionality of the structured latent state produced by StateEncoder."""

    # ── Decision Space ─────────────────────────────────────────────
    n_strategies: int = 3
    """Number of high-level strategies to choose from."""

    n_actions: int = 8
    """Number of low-level actions to choose from."""

    markov_rank: int = 16
    """Dimensionality of the action embedding in the RNN step."""

    # ── World Model ────────────────────────────────────────────────
    world_model_hidden: int = 64
    """Hidden layer size of the one-step world model."""

    # ── Gumbel Router (optional) ───────────────────────────────────
    router_hidden: int = 64
    """Hidden layer size of the GumbelSoftmax router (if used)."""

    # ── Domain Labels (for interpretability) ───────────────────────
    strategy_names: List[str] = field(
        default_factory=lambda: [f"strategy_{i}" for i in range(3)]
    )
    """Human-readable names for each strategy index."""

    action_names: List[str] = field(
        default_factory=lambda: [f"action_{i}" for i in range(8)]
    )
    """Human-readable names for each action index."""

    def __post_init__(self):
        """Validate configuration consistency."""
        assert self.state_dim > 0, "state_dim must be positive"
        assert self.n_strategies > 0, "n_strategies must be positive"
        assert self.n_actions > 0, "n_actions must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"

        if len(self.strategy_names) != self.n_strategies:
            self.strategy_names = [f"strategy_{i}" for i in range(self.n_strategies)]

        if len(self.action_names) != self.n_actions:
            self.action_names = [f"action_{i}" for i in range(self.n_actions)]


__all__ = ["GameNNConfig"]
